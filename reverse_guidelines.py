#!/usr/bin/env python3
"""Generate reversed (violating) guideline variants using gpt-5 via azure_gpt_call.

This script:
- Reads original guidelines from `guidelines/airlines/oracle.json`.
- For each guideline string, calls gpt-5 to synthesize 3–5 modified versions that
  intentionally violate the original rule while staying in-domain.
- Writes the result to `guidelines/modified.json` with the same key structure, but
  replacing each original string with a list of modified strings.

Category handling:
- Category 1 and 3: values are single strings; replaced by a list[str].
- Category 2: intent keys map to a list of phase strings; each phase string is
  replaced by a list[str] (handled step-by-step per phase).

Environment:
- Uses `azure_gpt_call.call_chat_completion` with model `gpt-5`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Iterable

from azure_gpt_call import call_chat_completion


class Progress:
    """Lightweight console progress visualization without external deps."""

    def __init__(self, total: int, *, width: int = 40, enable: bool | None = None) -> None:
        self.total = max(1, int(total))
        self.width = max(10, int(width))
        self.current = 0
        # Auto-enable when stdout is a TTY unless explicitly disabled
        self.enable = enable if enable is not None else sys.stdout.isatty()

    def _render_bar(self) -> str:
        ratio = min(1.0, max(0.0, self.current / self.total))
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        pct = int(ratio * 100)
        return f"[{bar}] {self.current}/{self.total} ({pct}%)"

    def update(self, inc: int = 1, detail: str | None = None) -> None:
        self.current = min(self.total, self.current + inc)
        if not self.enable:
            if detail:
                print(f"Progress {self.current}/{self.total}: {detail}")
            else:
                print(f"Progress {self.current}/{self.total}")
            return
        bar = self._render_bar()
        suffix = f"  {detail}" if detail else ""
        sys.stdout.write("\r" + bar + suffix + " " * 8)
        sys.stdout.flush()

    def done(self) -> None:
        if not self.enable:
            return
        sys.stdout.write("\n")
        sys.stdout.flush()


SYSTEM_PROMPT = (
    "You are an expert author of compliance adversarial examples for airline call center agent guidelines. "
    "Given one original guideline, you must produce several realistic but non-compliant alternative instructions that violate that guideline. "
    "Stay within the airline call center domain and keep outputs concise, actionable, and plausible as agent instructions. "
    "Return only a JSON array of strings with no commentary."
)


def _build_user_prompt(original: str, *, category: str, intent: str | None, phase_mode: bool) -> str:
    """Create an instruction explaining the modification intent and format."""
    scope_bits: list[str] = []
    scope_bits.append(f"Category: {category}")
    if intent:
        scope_bits.append(f"Intent: {intent}")
    if phase_mode:
        scope_bits.append("Treat this as one phase step only; do not reference other phases.")

    scope = " | ".join(scope_bits)
    guidance = (
        "Your task: Generate 3–5 modified versions of the SINGLE guideline below that intentionally violate the original.\n"
        "- Reflect realistic non-compliance (e.g., skip required steps, reverse constraints, leak private data, offer out-of-scope help, contradict phrasing, ignore verification, change tone to be non-compliant).\n"
        "- Keep variants specific to the original content and airline-contact-center context.\n"
        "- Keep each variant a standalone instruction line.\n"
        "- Do NOT explain; output ONLY a JSON array of 3–5 strings."
    )
    return (
        f"{scope}\n\n{guidance}\n\nOriginal guideline:\n" + json.dumps(original, ensure_ascii=False)
    )


def _extract_json_array(text: str) -> list[str] | None:
    """Attempt to extract a JSON array of strings from model output.

    Tries: direct parse; fenced code blocks; bracket slicing fallback.
    """
    text = text.strip()
    # Direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
    except Exception:
        pass

    # Code fence extraction
    fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                return parsed
        except Exception:
            pass

    # Bracket slicing fallback
    lb = text.find("[")
    rb = text.rfind("]")
    if lb != -1 and rb != -1 and rb > lb:
        snippet = text[lb : rb + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                return parsed
        except Exception:
            pass

    # Bullet fallback (last resort)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    bullets: list[str] = []
    for ln in lines:
        if re.match(r"^[-*\d]+[.)]?\s+", ln):
            bullets.append(re.sub(r"^[-*\d]+[.)]?\s+", "", ln))
    if 3 <= len(bullets) <= 5:
        return bullets
    return None


def generate_violations(original: str, *, category: str, intent: str | None = None, phase_mode: bool = False) -> list[str]:
    """Call gpt-5 to produce 3–5 violating variants for a single guideline string."""
    user_prompt = _build_user_prompt(original, category=category, intent=intent, phase_mode=phase_mode)
    messages = [{"role": "user", "content": user_prompt}]

    response_text = call_chat_completion(
        "gpt-5",
        messages,
        system_prompt=SYSTEM_PROMPT,
        reasoning_effort="medium",
    )
    variants = _extract_json_array(response_text) or []
    # Enforce bounds lightly; if outside, trim to 5 or pad by simple perturbations.
    if len(variants) > 5:
        variants = variants[:5]
    if len(variants) < 3:
        # Simple programmatic fallbacks to ensure minimum count
        base = original
        needed = 3 - len(variants)
        synthetics: list[str] = []
        # Create a few generic non-compliant tweaks as backup
        synthetics.append("Ignore verification and proceed without confirming caller identity.")
        synthetics.append("Disclose full payment card number aloud for convenience.")
        synthetics.append("Offer legal or medical advice unrelated to travel.")
        variants.extend(synthetics[:needed])
    return variants


def _count_units(data: dict[str, Any]) -> int:
    total = 0
    # Cat1
    cat1 = data.get("Category 1: Universal Compliance")
    if isinstance(cat1, dict):
        for v in cat1.values():
            if isinstance(v, str):
                total += 1
    # Cat2
    cat2 = data.get("Category 2: Intent Triggered Guidelines")
    if isinstance(cat2, dict):
        for phases in cat2.values():
            if isinstance(phases, list):
                for v in phases:
                    if isinstance(v, str):
                        total += 1
    # Cat3
    cat3 = data.get("Category 3: Condition Triggered Guidelines")
    if isinstance(cat3, dict):
        for v in cat3.values():
            if isinstance(v, str):
                total += 1
    return total


def process_oracle(data: dict[str, Any], *, show_progress: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {}
    total_units = _count_units(data)
    progress = Progress(total_units, enable=show_progress)

    # Category 1: flat dict[str, str]
    cat1 = data.get("Category 1: Universal Compliance")
    if isinstance(cat1, dict):
        out_cat1: dict[str, Any] = {}
        for key, guideline in cat1.items():
            if isinstance(guideline, str):
                detail = f"Cat1 {key}"
                out_cat1[key] = generate_violations(
                    guideline,
                    category="Category 1: Universal Compliance",
                )
                progress.update(detail=detail)
        out["Category 1: Universal Compliance"] = out_cat1

    # Category 2: intents -> list[str]
    cat2 = data.get("Category 2: Intent Triggered Guidelines")
    if isinstance(cat2, dict):
        out_cat2: dict[str, Any] = {}
        for intent, phases in cat2.items():
            if isinstance(phases, list):
                new_phases: list[Any] = []
                for phase_str in phases:
                    if isinstance(phase_str, str):
                        variants = generate_violations(
                            phase_str,
                            category="Category 2: Intent Triggered Guidelines",
                            intent=intent,
                            phase_mode=True,
                        )
                        new_phases.append(variants)
                        # Extract a brief "Phase X" tag for progress detail, if present
                        phase_label = "Phase"
                        m = re.match(r"^(Phase\s*\d+)\b", phase_str.strip())
                        if m:
                            phase_label = m.group(1)
                        progress.update(detail=f"Cat2 {intent} {phase_label}")
                out_cat2[intent] = new_phases
        out["Category 2: Intent Triggered Guidelines"] = out_cat2

    # Category 3: flat dict[str, str]
    cat3 = data.get("Category 3: Condition Triggered Guidelines")
    if isinstance(cat3, dict):
        out_cat3: dict[str, Any] = {}
        for key, guideline in cat3.items():
            if isinstance(guideline, str):
                detail = f"Cat3 {key}"
                out_cat3[key] = generate_violations(
                    guideline, category="Category 3: Condition Triggered Guidelines"
                )
                progress.update(detail=detail)
        out["Category 3: Condition Triggered Guidelines"] = out_cat3

    progress.done()
    return out


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthesize violating guidelines using gpt-5")
    parser.add_argument(
        "--input",
        default=os.path.join("guidelines", "airlines", "oracle.json"),
        help="Path to the original oracle.json",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("guidelines", "modified.json"),
        help="Path to write the modified guidelines JSON",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable console progress visualization",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    with open(args.input, "r", encoding="utf-8") as f:
        oracle = json.load(f)

    modified = process_oracle(oracle, show_progress=(not args.no_progress))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(modified, f, ensure_ascii=False, indent=2)

    print(f"\nWrote modified guidelines to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
