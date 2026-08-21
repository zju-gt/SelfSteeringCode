"""Generate one answer with a saved Router and primitive library."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--router", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-cache-routing", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from riser.inference import SteeredModel
    from riser.router import RouterInference

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(args.device)
    routing = RouterInference.from_pretrained(
        args.router,
        args.library,
        target_layer=args.layer,
        device=args.device,
    )
    steered = SteeredModel(
        model,
        routing,
        cache_routing=not args.no_cache_routing,
    )
    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    output = steered.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated_ids = output[0, inputs.input_ids.shape[1]:]
    print(tokenizer.decode(generated_ids, skip_special_tokens=True))
    print("routing:", steered.get_last_routing_info())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
