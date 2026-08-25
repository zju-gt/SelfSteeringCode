"""Evaluate baseline and fixed/Router-steered generation on MMLU JSONL."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from riser.evaluation.cli import load_examples
from riser.evaluation.metrics import exact_match, substring_match
from riser.evaluation.paths import resolve_results_path
from riser.evaluation.runner import EvaluationRunner
from riser.inference import SteeredModel
from riser.router import RouterInference

from scripts.prepare_mmlu_eval import mmlu_choice_match


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model name/path")
    parser.add_argument(
        "--router",
        help="Router checkpoint path; omit it to use deterministic fixed routing",
    )
    parser.add_argument("--library", required=True, help="Primitive library .pt path")
    parser.add_argument("--layer", required=True, type=int, help="Injection layer")
    parser.add_argument("--input", required=True, help="MMLU evaluation JSONL")
    parser.add_argument("--output", help="Results JSONL; defaults to a timestamped path")
    parser.add_argument(
        "--output-dir",
        default="artifacts/mmlu_preliminary",
        help="Directory for the default timestamped results file",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of examples to generate in each Transformers batch",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--fixed-primitives",
        type=int,
        nargs="+",
        help="Primitive IDs for no-checkpoint mode",
    )
    parser.add_argument(
        "--fixed-strengths",
        type=float,
        nargs="+",
        help="Strength for each fixed primitive; defaults to 1.0",
    )
    parser.add_argument("--fixed-max-strength", type=float, default=2.0)
    parser.add_argument("--no-cache-routing", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar during benchmark inference",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.router is None and args.fixed_strengths is not None and args.fixed_primitives is None:
        raise ValueError("--fixed-strengths requires --fixed-primitives")
    if args.fixed_primitives is not None and args.fixed_strengths is not None:
        if len(args.fixed_primitives) != len(args.fixed_strengths):
            raise ValueError("--fixed-primitives and --fixed-strengths must have equal lengths")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    output_path = resolve_results_path(args.output, args.output_dir)
    dtype = getattr(torch, args.dtype)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    )
    model.to(args.device)
    model.eval()

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
        model,
        routing,
        cache_routing=not args.no_cache_routing,
    )
    runner = EvaluationRunner(
        baseline_model=model,
        steered_model=steered_model,
        tokenizer=tokenizer,
        device=args.device,
    )
    result_count = runner.run_to_jsonl(
        load_examples(args.input),
        output_path,
        generation_kwargs={
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
        },
        metrics={
            "mmlu_accuracy": mmlu_choice_match,
            "exact_match": exact_match,
            "substring_match": substring_match,
        },
        batch_size=args.batch_size,
        show_progress=not args.no_progress,
        progress_desc="MMLU inference",
        result_metadata={
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "layer": args.layer,
            "primitive": args.fixed_primitives or [],
            "strength": args.fixed_strengths or [],
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    print(f"Wrote {result_count} MMLU evaluation results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
