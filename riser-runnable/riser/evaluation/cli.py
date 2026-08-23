"""Command-line helpers for the initial RISER evaluation flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from .metrics import exact_match, substring_match
from .records import EvaluationExample
from .runner import EvaluationRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a baseline Hugging Face model with RISER steering."
    )
    parser.add_argument("--model", required=True, help="Hugging Face model name or path")
    parser.add_argument(
        "--router",
        help="Router checkpoint path; omit it to use deterministic fixed routing",
    )
    parser.add_argument("--library", required=True, help="Primitive library .pt path")
    parser.add_argument("--layer", required=True, type=int, help="Target transformer layer")
    parser.add_argument("--input", required=True, help="Input JSONL examples")
    parser.add_argument("--output", required=True, help="Output JSONL results")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--fixed-primitives",
        type=int,
        nargs="+",
        help="Primitive IDs for no-checkpoint mode, e.g. --fixed-primitives 0 2",
    )
    parser.add_argument(
        "--fixed-strengths",
        type=float,
        nargs="+",
        help="Strength for each fixed primitive; defaults to 1.0",
    )
    parser.add_argument("--fixed-max-strength", type=float, default=2.0)
    parser.add_argument(
        "--no-cache-routing",
        action="store_true",
        help="Recompute routing on every forward instead of reusing the prefill route",
    )
    return parser


def load_examples(path) -> List[EvaluationExample]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation input not found: {path}")
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                values = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            examples.append(EvaluationExample.from_dict(values))
    return examples


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.router is None and args.fixed_strengths is not None and args.fixed_primitives is None:
        raise ValueError("--fixed-strengths requires --fixed-primitives")
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    )
    base_model.to(args.device)

    from ..inference import SteeredModel
    from ..router import RouterInference

    routing = RouterInference.from_pretrained(
        router_path=args.router,
        primitive_library_path=args.library,
        target_layer=args.layer,
        device=args.device,
        fixed_primitives=args.fixed_primitives,
        fixed_strengths=args.fixed_strengths,
        fixed_max_strength=args.fixed_max_strength,
    )
    steered_model = SteeredModel(
        base_model,
        routing,
        cache_routing=not args.no_cache_routing,
    )
    runner = EvaluationRunner(
        baseline_model=base_model,
        steered_model=steered_model,
        tokenizer=tokenizer,
        device=args.device,
    )
    results = runner.run(
        load_examples(args.input),
        generation_kwargs={
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
        },
        metrics={"exact_match": exact_match, "substring_match": substring_match},
    )
    runner.write_jsonl(results, args.output)
    print(f"Wrote {len(results)} evaluation results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
