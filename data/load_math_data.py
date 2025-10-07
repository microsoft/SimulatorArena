#!/usr/bin/env python3
"""
Restore redacted math tutoring annotations by loading problem files.

Usage:
  python load_math_data.py [REDACTED_JSON] [MATH_ROOT] [-o OUTPUT]
  python load_math_data.py --all [MATH_ROOT]  # Restore both redacted files

Positional args (optional):
  REDACTED_JSON   Path to the redacted annotations JSON (default: math_tutoring_annotations_redacted.json)
  MATH_ROOT       Root folder containing the MATH files (default: MATH)

Options:
  -o, --output    Output JSON path (default: math_tutoring_annotations.json)
  --all           Restore both redacted files (main and benchmarking)
  --no-strict     Disable strict checks for required keys in problem files

Example:
  python load_math_data.py  # Restore main file using defaults
  python load_math_data.py --all  # Restore both files using defaults
  python load_math_data.py custom_redacted.json  # Custom redacted file, default MATH root
  python load_math_data.py math_tutoring_annotations_redacted.json /path/to/MATH  # Custom paths
"""

import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def _resolve_path(location: str, root: Optional[Path]) -> Path:
    """Resolve a location string to an absolute Path, using root for relative paths."""
    loc_path = Path(location)
    if loc_path.is_absolute():
        return loc_path
    if root is None:
        return (Path.cwd() / loc_path).resolve()
    return (root / loc_path).resolve()


def _load_math_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Location not found on disk: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


_BOXED_RE = re.compile(r"\\boxed\s*\{(.*?)\}", flags=re.DOTALL)

import re
from typing import Optional

def _extract_boxed_answer(tex: Optional[str]) -> Optional[str]:
    """Return the last \\boxed{...} content with balanced-brace parsing."""
    if not isinstance(tex, str):
        return None

    # Find the last occurrence of '\boxed{'
    last_open_brace_idx = None
    for m in re.finditer(r'\\boxed\s*\{', tex):
        last_open_brace_idx = m.end() - 1  # position of the '{'

    if last_open_brace_idx is None:
        return None

    i = last_open_brace_idx + 1  # first char inside the braces
    depth = 1
    out = []

    while i < len(tex) and depth > 0:
        ch = tex[i]
        prev = tex[i - 1] if i > 0 else ''
        if ch == '{' and prev != '\\':
            depth += 1
            out.append(ch)
        elif ch == '}' and prev != '\\':
            depth -= 1
            if depth == 0:
                break  # found the matching closing brace for the \boxed{...}
            out.append(ch)
        else:
            out.append(ch)
        i += 1

    ans = ''.join(out).strip()

    # If the entire content is wrapped in $...$, strip those outer $ signs
    if len(ans) >= 2 and ans[0] == '$' and ans[-1] == '$':
        ans = ans[1:-1].strip()

    # Do NOT strip trailing '.' here; the period usually lives outside \boxed{...}
    return ans or None



def _restore_one(
    ann: Dict[str, Any],
    dataset_root: Optional[Path],
    strict: bool = True,
) -> Dict[str, Any]:
    if "question_location" not in ann or "similar_question_location" not in ann:
        raise ValueError("Annotation missing 'question_location' or 'similar_question_location'")

    q_path = _resolve_path(str(ann["question_location"]), dataset_root)
    s_path = _resolve_path(str(ann["similar_question_location"]), dataset_root)

    q_obj = _load_math_json(q_path)
    s_obj = _load_math_json(s_path)

    if strict:
        for pth, obj in ((q_path, q_obj), (s_path, s_obj)):
            for key in ("problem", "solution"):
                if key not in obj:
                    raise KeyError(f"{pth} missing key '{key}'")

    q_problem = q_obj.get("problem")
    q_solution = q_obj.get("solution")
    s_problem = s_obj.get("problem")
    s_solution = s_obj.get("solution")

    out = copy.deepcopy(ann)
    # Restore original fields
    out["question"] = q_problem
    out["solution"] = q_solution
    out["similar_question"] = s_problem
    out["problem_1_gold_final_answer"] = _extract_boxed_answer(q_solution)
    out["problem_2_gold_solution"] = s_solution
    out["problem_2_gold_final_answer"] = _extract_boxed_answer(s_solution)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore redacted annotations using MATH problem files."
    )
    parser.add_argument(
        "redacted_json",
        type=Path,
        nargs='?',
        default=Path("math_tutoring_annotations_redacted.json"),
        help="Path to redacted annotations JSON (default: math_tutoring_annotations_redacted.json)"
    )
    parser.add_argument(
        "math_root",
        type=Path,
        nargs='?',
        default=Path("MATH"),
        help="Root folder of MATH dataset files (default: MATH)"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: math_tutoring_annotations.json)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Restore both main and benchmarking redacted files",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Disable strict key checks while loading problem files",
    )
    args = parser.parse_args()

    # If --all flag is used, restore both files
    if args.all:
        files_to_process = [
            (Path("math_tutoring_annotations_redacted.json"),
             Path("math_tutoring_annotations.json")),
            (Path("math_tutoring_annotations_redacted_for_benchmarking.json"),
             Path("math_tutoring_annotations_for_benchmarking.json"))
        ]
    else:
        # Single file mode
        output_path = args.output if args.output else Path("math_tutoring_annotations.json")
        files_to_process = [(args.redacted_json, output_path)]

    dataset_root = args.math_root.resolve()
    strict = not args.no_strict

    for redacted_path, output_path in files_to_process:
        print(f"\nProcessing: {redacted_path} -> {output_path}")
        _restore_file(redacted_path, output_path, dataset_root, strict)


def _restore_file(redacted_json: Path, output_path: Path, dataset_root: Path, strict: bool) -> None:
    """Restore a single redacted JSON file."""

    # Load redacted input
    try:
        with redacted_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Error reading redacted JSON: {e}", file=sys.stderr)
        sys.exit(1)

    # Accept either a list[...] or {"annotations": [...]}
    if isinstance(data, list):
        redacted_list = data
        wrap_in_dict = False
    elif isinstance(data, dict) and "annotations" in data and isinstance(data["annotations"], list):
        redacted_list = data["annotations"]
        wrap_in_dict = True
    else:
        print(
            "  Input must be a list of annotations or a dict with key 'annotations' (list).",
            file=sys.stderr,
        )
        sys.exit(2)

    restored_list: List[Dict[str, Any]] = []
    for i, ann in enumerate(redacted_list):
        try:
            restored = _restore_one(ann, dataset_root=dataset_root, strict=strict)
            restored_list.append(restored)
        except Exception as e:
            print(f"  [index {i}] Error restoring annotation: {e}", file=sys.stderr)
            sys.exit(3)

    # Preserve original top-level structure
    out_obj: Union[List[Dict[str, Any]], Dict[str, Any]] = (
        {"annotations": restored_list} if wrap_in_dict else restored_list
    )

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic-ish write
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        tmp_path.replace(output_path)
    except Exception as e:
        print(f"  Failed to write output: {e}", file=sys.stderr)
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        sys.exit(4)

    print(f"  ✓ Wrote restored annotations to: {output_path}")


if __name__ == "__main__":
    main()