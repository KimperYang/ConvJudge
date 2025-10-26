#!/usr/bin/env python3
"""
Normalize mistake category labels in generated conversation data so they match
the canonical guideline keys.

Usage:
  python scripts/normalize_labels.py --dry-run
  python scripts/normalize_labels.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple


def load_guideline_metadata(
    paths: Iterable[pathlib.Path],
) -> Tuple[Set[str], Dict[str, str], Dict[str, str]]:
    """
    Build the set of canonical guideline keys, a mapping of aliases to the
    canonical label, and a lookup from descriptive text back to the label.
    """
    canonical: Set[str] = set()
    aliases: Dict[str, str] = {}
    text_lookup: Dict[str, str] = {}

    def register_alias(alias: str, canonical_key: str) -> None:
        alias = alias.strip()
        canonical_key = canonical_key.strip()
        if not alias or not canonical_key:
            return
        aliases.setdefault(alias, canonical_key)

    category_prefix = re.compile(r"^Category\s+\d+:\s*")

    def walk(obj, parent: Optional[str] = None) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(key, str):
                    key_str = key.strip()
                    if key_str:
                        canonical.add(key_str)
                        register_alias(key_str, key_str)
                        if parent:
                            combined = f"{parent} :: {key_str}"
                            text_lookup.setdefault(combined, key_str)
                            if match := category_prefix.match(parent):
                                parent_short = parent[match.end():].strip()
                                if parent_short:
                                    combined_short = f"{parent_short} :: {key_str}"
                                    text_lookup.setdefault(combined_short, key_str)
                walk(value, key)
        elif isinstance(obj, list):
            for item in obj:
                walk(item, parent)
        elif isinstance(obj, str):
            key_str = obj.strip()
            if not key_str:
                return
            canonical.add(key_str)
            register_alias(key_str, key_str)
            if ":" in key_str:
                prefix, suffix = key_str.split(":", 1)
                prefix = prefix.strip()
                suffix = suffix.strip()
                if prefix:
                    register_alias(prefix, key_str)
                if suffix:
                    register_alias(suffix, key_str)
            if parent:
                parent_key = parent.strip() if isinstance(parent, str) else ""
                if parent_key:
                    text_lookup.setdefault(key_str, parent_key)
            # Register variants without parenthetical hints.
            no_paren = re.sub(r"\s*\([^)]*\)", "", key_str).strip()
            if no_paren and no_paren != key_str:
                register_alias(no_paren, key_str)
                if ":" in no_paren:
                    prefix_np, suffix_np = no_paren.split(":", 1)
                    prefix_np = prefix_np.strip()
                    suffix_np = suffix_np.strip()
                    if prefix_np:
                        register_alias(prefix_np, key_str)
                    if suffix_np:
                        register_alias(suffix_np, key_str)
            # Register trailing components after dashes.
            for sep in ("—", "-", "‑"):
                if sep in key_str:
                    tail = key_str.split(sep, 1)[1].strip()
                    if tail:
                        register_alias(tail, key_str)
                        if ":" in tail:
                            head, rest = tail.split(":", 1)
                            head = head.strip()
                            rest = rest.strip()
                            if head:
                                register_alias(head, key_str)
                            if rest:
                                register_alias(rest, key_str)

    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        walk(data)

    return canonical, aliases, text_lookup


def generate_candidates(label: str) -> List[str]:
    label = label.strip()
    if not label:
        return []

    variants: List[str] = [label]
    category_prefix = re.compile(r"^Category\s+\d+:\s*")

    # Remove "Category X:" prefixes.
    if category_prefix.match(label):
        variants.append(category_prefix.sub("", label).strip())

    # Split on "::", "/", ":" to extract subcomponents.
    if "::" in label:
        variants.extend(part.strip() for part in label.split("::") if part.strip())
    if "/" in label:
        variants.extend(part.strip() for part in label.split("/") if part.strip())
    if ":" in label:
        prefix, suffix = label.split(":", 1)
        variants.append(prefix.strip())
        variants.append(suffix.strip())

    # Handle hyphen variants.
    hyphen_variants = {
        label.replace(" - ", " — "),
        label.replace(" — ", " - "),
        label.replace("-", " — "),
        label.replace("—", "-"),
        label.replace("‑", "-"),
        label.replace("-", "‑"),
    }
    for variant in hyphen_variants:
        variants.append(variant.strip())

    for sep in ("—", "-", "‑"):
        if sep in label:
            variants.extend(part.strip() for part in label.split(sep) if part.strip())

    # Deduplicate while preserving order.
    seen: Set[str] = set()
    result: List[str] = []
    for variant in variants:
        if variant and variant not in seen:
            seen.add(variant)
            result.append(variant)
    return result


def resolve_category(
    category: Optional[str],
    guideline_key: Optional[str],
    canonical: Set[str],
    aliases: Dict[str, str],
    text_lookup: Dict[str, str],
) -> Optional[str]:
    candidates: List[str] = []

    if category:
        candidates.extend(generate_candidates(category))
    if guideline_key:
        candidates.extend(generate_candidates(guideline_key))
        key_trim = guideline_key.strip()
        if key_trim:
            candidates.append(key_trim)

    # Allow mapping via descriptive text (value -> label).
    for candidate in list(candidates):
        mapped = text_lookup.get(candidate)
        if mapped:
            candidates.append(mapped)

    for candidate in candidates:
        if candidate in canonical:
            return candidate
        if candidate in aliases:
            return aliases[candidate]

    for candidate in candidates:
        mapped = text_lookup.get(candidate)
        if mapped:
            if mapped in canonical:
                return mapped
            if mapped in aliases:
                return aliases[mapped]

    return None


def normalize_categories(
    sample: Dict,
    canonical: Set[str],
    aliases: Dict[str, str],
    text_lookup: Dict[str, str],
) -> Tuple[bool, List[str]]:
    """
    Normalize category labels within a single conversation sample.
    Returns (changed, errors).
    """
    changed = False
    errors: List[str] = []

    def fix_entry(entry: Dict, location: str) -> None:
        nonlocal changed
        if not isinstance(entry, dict):
            return
        category = entry.get("category")
        if not category:
            return
        guideline_key = entry.get("guideline_key")
        resolved = resolve_category(category, guideline_key, canonical, aliases, text_lookup)
        if resolved:
            if entry.get("category") != resolved:
                entry["category"] = resolved
                changed = True
        else:
            errors.append(f"{location}: {category}")

    for idx, item in enumerate(sample.get("mistakes", [])):
        fix_entry(item, f"mistakes[{idx}]")

    meta = sample.get("meta")
    if isinstance(meta, dict):
        for idx, item in enumerate(meta.get("mistakes_planned", [])):
            fix_entry(item, f"meta.mistakes_planned[{idx}]")

    return changed, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize mistake categories to guideline keys.")
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=pathlib.Path("data"),
        help="Directory containing generated conversation JSON files.",
    )
    parser.add_argument(
        "--guidelines",
        type=pathlib.Path,
        action="append",
        default=[pathlib.Path("guidelines/airlines/oracle.json")],
        help="Path(s) to guideline JSON files defining canonical labels.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without writing back to disk.",
    )
    args = parser.parse_args()

    canonical, aliases, text_lookup = load_guideline_metadata(args.guidelines)

    data_files = sorted(args.data_dir.glob("*.json"))
    if not data_files:
        raise SystemExit(f"No JSON files found in {args.data_dir}")

    unresolved: Dict[pathlib.Path, List[str]] = {}
    changed_files: List[pathlib.Path] = []

    for path in data_files:
        with path.open("r", encoding="utf-8") as fh:
            sample = json.load(fh)

        changed, errors = normalize_categories(sample, canonical, aliases, text_lookup)
        if errors:
            unresolved[path] = errors

        if changed and not args.dry_run:
            with path.open("w", encoding="utf-8") as fh:
                json.dump(sample, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
            changed_files.append(path)
        elif changed:
            changed_files.append(path)

    if changed_files:
        print("Normalized categories in:")
        for path in changed_files:
            print(f"  - {path}")
    else:
        print("No category changes needed.")

    if unresolved:
        print("\nUnresolved categories:")
        for path, issues in unresolved.items():
            print(f"- {path}")
            for issue in issues:
                print(f"    {issue}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
