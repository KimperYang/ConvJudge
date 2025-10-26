#!/usr/bin/env python3
"""Evaluate guideline violations detected by an LLM against labeled conversations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from azure_gpt_call import MissingConfiguration, call_chat_completion
from json import JSONDecodeError


SYSTEM_PROMPT = (
    "You are a meticulous compliance adjudicator for Celestar Air's virtual assistant. "
    "Given the official guidelines and a fully transcribed conversation between a caller "
    "and the agent, identify every assistant turn that violates a guideline. "
    "Only mark a violation when the guideline is clearly broken. "
    "Return strictly the JSON structure requested—no surrounding prose."
)


@dataclass(frozen=True)
class ViolationKey:
    """Canonical representation of a violation for comparison."""

    turn_index: int
    guideline_type: str
    guideline_phase: int

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ViolationKey":
        turn_index = int(payload["turn_index"])
        guideline_type = str(payload["guideline_type"]).strip()
        guideline_phase_raw = payload.get("guideline_phase", -1)
        try:
            guideline_phase = int(guideline_phase_raw)
        except (TypeError, ValueError):
            guideline_phase = -1
        return cls(turn_index, guideline_type, guideline_phase)


def _format_guideline_bucket(bucket: dict[str, Any]) -> str:
    """Render a mapping of guideline types to their descriptions."""
    lines: list[str] = []
    for guideline_type, description in bucket.items():
        lines.append(f"- Guideline type: {guideline_type}\n  Description: {description}")
    return "\n".join(lines) if lines else "(none)"


def format_guidelines(guidelines: dict[str, Any], intent: str | None) -> str:
    """Format guidelines into a readable textual block for the prompt."""
    category1 = guidelines.get("Category 1: Universal Compliance / General Knowledge", {}) or {}
    category3 = guidelines.get("Category 3: Condition Triggered Guidelines", {}) or {}
    category2 = guidelines.get("Category 2: Intent Triggered Guidelines", {}) or {}

    general_block = _format_guideline_bucket(category1)
    conditional_block = _format_guideline_bucket(category3)

    if intent and isinstance(category2, dict) and intent in category2:
        intent_steps = category2[intent]
        if isinstance(intent_steps, Sequence):
            intent_lines = [f"{step}" for idx, step in enumerate(intent_steps)]
            intent_block = "\n".join(intent_lines)
        else:
            intent_block = json.dumps(intent_steps, indent=2, ensure_ascii=False)
    else:
        intent_block = json.dumps(category2, indentc=2, ensure_ascii=False)

    formatted = (
        "GENERAL GUIDELINES (use the guideline_type exactly as the key shown):\n"
        f"{general_block}\n\n"
        "CONDITION TRIGGERED GUIDELINES (use the guideline_type exactly as the key shown):\n"
        f"{conditional_block}\n\n"
        f"INTENT WORKFLOW (guideline_type must be '{intent}' when citing these phases):\n"
        f"{intent_block}"
    )
    return formatted


def format_conversation(conversation: Sequence[dict[str, Any]]) -> str:
    """Pretty-print the conversation for the prompt."""
    lines: list[str] = []
    for turn in conversation:
        turn_index = turn.get("turn_index", len(lines))
        role = turn.get("role", "assistant")
        content = turn.get("content", "")
        lines.append(f"{turn_index} | {role.upper()}: {content}")
    return "\n".join(lines)


def build_user_prompt(guidelines_text: str, conversation_text: str, conversation_id: str) -> str:
    """Compose the user prompt that instructs the model."""
    instructions = (
        "TASK:\n"
        "Analyze the conversation transcript using the provided guidelines. "
        "Identify every assistant (agent) turn that violates a guideline. "
        "Use turn indices exactly as shown. "
        "When reporting a violation, set guideline_type to the exact guideline key provided in the reference text "
        "(for the intent workflow, always use the intent name itself). "
        "For general guidelines (Category 1 or Category 3) set guideline_phase to -1. "
        "For intent workflow violations, set guideline_phase to corresponding Phase number (only include number here without Phase text prefix).\n"
        "RESPONSE FORMAT (strict JSON, no extra text):\n"
        '{\n'
        f'  "conversation_id": "{conversation_id}",\n'
        '  "violations": [\n'
        '    {\n'
        '      "turn_index": <int>,\n'
        '      "guideline_type": "<string>",\n'
        '      "guideline_phase": <int>,\n'
        '      "evidence": "<short quote from the assistant message>",\n'
        '      "explanation": "<why the guideline is violated>"\n'
        '    }\n'
        '  ]\n'
        '}\n\n'
        "Return an empty list if you find no violations. Do not include any commentary outside the JSON object."
    )
    prompt = (
        f"{instructions}\n\n"
        "GUIDELINES REFERENCE:\n"
        f"{guidelines_text}\n\n"
        "CONVERSATION TRANSCRIPT:\n"
        f"{conversation_text}"
    )
    return prompt


def extract_json(response_text: str) -> dict[str, Any]:
    """Extract and parse the first JSON object found in the model response."""
    text = response_text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            code_block = parts[1]
            if code_block.startswith("json"):
                code_block = code_block[len("json") :]
            text = code_block.strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


def repair_response_json(raw_response: str, model: str, conversation_id: str) -> dict[str, Any]:
    """Ask the LLM to rewrite its own response into valid JSON."""
    instructions = (
        "You are a strict JSON formatter. Rewrite the provided content so it becomes valid JSON that matches "
        "this structure exactly:\n"
        "{\n"
        '  "conversation_id": "<string>",\n'
        '  "violations": [\n'
        "    {\n"
        '      "turn_index": <int>,\n'
        '      "guideline_type": "<string>",\n'
        '      "guideline_phase": <int>,\n'
        '      "evidence": "<string>",\n'
        '      "explanation": "<string>"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Preserve the original semantic content and conversation_id. Remove any commentary or markdown. "
        "Return only the corrected JSON object."
    )
    messages = [
        {"role": "system", "content": instructions},
        {
            "role": "user",
            "content": f"conversation_id: {conversation_id}\nOriginal response:\n{raw_response}",
        },
    ]
    repaired_text = call_chat_completion(model, messages)
    return extract_json(repaired_text)


def ensure_output_dir(path: Path) -> None:
    """Create the output directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def compute_confusion(
    predicted: Iterable[ViolationKey], reference: Iterable[ViolationKey]
) -> tuple[list[ViolationKey], list[ViolationKey], list[ViolationKey]]:
    """Return (true_positive, false_positive, false_negative) lists."""
    predicted_set = set(predicted)
    reference_set = set(reference)
    true_positive = sorted(predicted_set & reference_set, key=lambda v: (v.turn_index, v.guideline_type))
    false_positive = sorted(predicted_set - reference_set, key=lambda v: (v.turn_index, v.guideline_type))
    false_negative = sorted(reference_set - predicted_set, key=lambda v: (v.turn_index, v.guideline_type))
    return true_positive, false_positive, false_negative


def compute_precision_recall(tp: int, fp: int, fn: int) -> tuple[float, float]:
    """Compute precision and recall with zero-division safeguards."""
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return precision, recall


def violation_list_to_json(keys: Iterable[ViolationKey]) -> list[dict[str, Any]]:
    """Convert violation keys back into JSON-serializable objects."""
    return [
        {
            "turn_index": key.turn_index,
            "guideline_type": key.guideline_type,
            "guideline_phase": key.guideline_phase,
        }
        for key in keys
    ]


def evaluate_sample(
    sample_path: Path,
    guidelines: dict[str, Any],
    model: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate a single conversation file."""
    with sample_path.open("r", encoding="utf-8") as handle:
        sample = json.load(handle)

    conversation_id = sample.get("meta", {}).get("id") or sample_path.stem
    intent = sample.get("meta", {}).get("intent")
    conversation = sample.get("conversation", [])
    reference_mistakes = sample.get("mistakes", [])

    guidelines_text = format_guidelines(guidelines, intent)
    conversation_text = format_conversation(conversation)
    user_prompt = build_user_prompt(guidelines_text, conversation_text, conversation_id)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    response_text = call_chat_completion(model, messages)
    try:
        response_json = extract_json(response_text)
    except JSONDecodeError:
        print(f"Initial parsing failed for {conversation_id}; attempting JSON repair.")
        try:
            response_json = repair_response_json(response_text, model, conversation_id)
        except JSONDecodeError as repair_exc:
            raise ValueError(
                "Failed to repair model response into valid JSON."
            ) from repair_exc

    predicted_payload = response_json.get("violations", []) or []
    predicted_keys = [ViolationKey.from_mapping(item) for item in predicted_payload]
    reference_keys = [ViolationKey.from_mapping(item) for item in reference_mistakes]

    true_positive, false_positive, false_negative = compute_confusion(predicted_keys, reference_keys)
    precision, recall = compute_precision_recall(len(true_positive), len(false_positive), len(false_negative))

    result_payload = {
        "conversation_file": str(sample_path),
        "conversation_id": conversation_id,
        "model": model,
        "predicted": predicted_payload,
        "ground_truth": reference_mistakes,
        "true_positive": violation_list_to_json(true_positive),
        "false_positive": violation_list_to_json(false_positive),
        "false_negative": violation_list_to_json(false_negative),
        "precision": precision,
        "recall": recall,
        "model_response_text": response_text,
    }

    output_path = output_dir / f"{sample_path.stem}_evaluation.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2, ensure_ascii=False)

    return result_payload


def summarize_metrics(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compute macro-averaged precision and recall."""
    precisions = [result["precision"] for result in results]
    recalls = [result["recall"] for result in results]

    def safe_average(values: Sequence[float]) -> float:
        if not values:
            return math.nan
        return sum(values) / len(values)

    macro_precision = safe_average(precisions)
    macro_recall = safe_average(recalls)

    summary = {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "num_samples": len(precisions),
    }
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate guideline violations using an LLM.")
    parser.add_argument(
        "--model",
        default=os.environ.get("AZURE_OPENAI_DEPLOYMENT") or os.environ.get("OPENAI_MODEL", "gpt-4.1"),
        help="Model/deployment name to use for evaluation.",
    )
    parser.add_argument(
        "--guidelines",
        default="guidelines/airlines/oracle.json",
        help="Path to the oracle guidelines JSON file.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing labeled conversation JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="dump/evaluation",
        help="Directory to write per-sample evaluation JSON files.",
    )
    parser.add_argument(
        "--summary-csv",
        default="evaluation_summary.csv",
        help="Filename for the aggregated metrics CSV (stored in the run-specific output directory).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    guidelines_path = Path(args.guidelines)
    data_dir = Path(args.data_dir)
    base_output_dir = Path(args.output_dir)

    if not guidelines_path.exists():
        raise FileNotFoundError(f"Guidelines file not found: {guidelines_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    ensure_output_dir(base_output_dir)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_name = str(args.model or "").strip() or "model"
    sanitized_model = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in model_name)
    run_dir = base_output_dir / f"{sanitized_model}_{timestamp}"
    ensure_output_dir(run_dir)
    summary_csv_path = run_dir / args.summary_csv

    with guidelines_path.open("r", encoding="utf-8") as handle:
        guidelines = json.load(handle)

    conversation_files = sorted(data_dir.glob("*.json"))
    if not conversation_files:
        print(f"No conversation files found in {data_dir}")
        return 0

    results: list[dict[str, Any]] = []
    for sample_path in conversation_files:
        try:
            result = evaluate_sample(sample_path, guidelines, args.model, run_dir)
            results.append(result)
            print(
                f"Evaluated {sample_path.name}: precision={result['precision']:.2f}, "
                f"recall={result['recall']:.2f}"
            )
        except MissingConfiguration as exc:
            print(f"Configuration error while evaluating {sample_path}: {exc}")
            return 2
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Failed to evaluate {sample_path}: {exc}")
            return 3

    summary = summarize_metrics(results)
    totals = {
        "tp": sum(len(result["true_positive"]) for result in results),
        "fp": sum(len(result["false_positive"]) for result in results),
        "fn": sum(len(result["false_negative"]) for result in results),
    }

    csv_columns = ["conversation_file", "conversation_id", "precision", "recall", "tp", "fp", "fn"]
    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "conversation_file": result["conversation_file"],
                    "conversation_id": result["conversation_id"],
                    "precision": f"{result['precision']:.6f}",
                    "recall": f"{result['recall']:.6f}",
                    "tp": len(result["true_positive"]),
                    "fp": len(result["false_positive"]),
                    "fn": len(result["false_negative"]),
                }
            )
        writer.writerow(
            {
                "conversation_file": "AVERAGE",
                "conversation_id": "",
                "precision": "" if math.isnan(summary["macro_precision"]) else f"{summary['macro_precision']:.6f}",
                "recall": "" if math.isnan(summary["macro_recall"]) else f"{summary['macro_recall']:.6f}",
                "tp": totals["tp"],
                "fp": totals["fp"],
                "fn": totals["fn"],
            }
        )

    print(
        f"Macro precision={summary['macro_precision']:.2f}, "
        f"macro recall={summary['macro_recall']:.2f} "
        f"across {summary['num_samples']} sample(s). Summary CSV: {summary_csv_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
