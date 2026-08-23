"""Build a primitive library from user-provided positive/negative prompt pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riser.primitives import (
    ActivationExtractor,
    PrimitiveClustering,
    PrimitiveLibrary,
)


def load_pairs(path):
    pairs = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            values = json.loads(line)
            pairs.append(
                (
                    values["positive_prompt"],
                    values["negative_prompt"],
                    values.get("task", ""),
                    str(values.get("task_id", values.get("id", line_number))),
                )
            )
    return pairs


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Prompt-pair JSONL")
    parser.add_argument("--model", required=True, help="Hugging Face model name/path")
    parser.add_argument("--output", required=True, help="Primitive library .pt output")
    parser.add_argument("--metadata-output")
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--aggregation", choices=["last", "concat", "mean"], default="last")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    prompt_pairs = load_pairs(args.input)
    extractor = ActivationExtractor(
        model_name=args.model,
        target_layers=args.layers,
        device=args.device,
    )
    activation_pairs = extractor.extract_batch(
        prompt_pairs,
        max_length=args.max_length,
        layer_aggregation=args.aggregation,
    )
    clustering = PrimitiveClustering(n_clusters=args.clusters)
    primitives = clustering.fit(activation_pairs)
    library = PrimitiveLibrary(
        primitives=primitives,
        metadata={
            "source": str(args.input),
            "model": args.model,
            "target_layers": args.layers,
            "num_pairs": len(activation_pairs),
        },
    )
    library.save(args.output, metadata_path=args.metadata_output)
    print(f"Saved {library.num_primitives} primitives to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
