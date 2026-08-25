"""Collect contrastive activation vectors from 500 MMLU mathematics items.

The script uses RISER's proof-level positive prompt and plausible-but-
unverified negative prompt to contrast reasoning fidelity.  It
then extracts the final-token activations for both prompts, saves the raw
positive/negative/difference tensors, and clusters the differences into a
primitive library that can be used by the steering examples.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riser.primitives import ActivationExtractor, PrimitiveClustering, PrimitiveLibrary


LOGGER = logging.getLogger(__name__)
DEFAULT_SUBJECTS = (
    "abstract_algebra",
    "college_mathematics",
    "elementary_mathematics",
    "high_school_mathematics",
)


def format_choices(choices) -> str:
    return "\n".join(
        f"{chr(ord('A') + index)}. {choice}"
        for index, choice in enumerate(choices)
    )


def make_prompt_pair(subject: str, question: str, choices) -> tuple[str, str, str]:
    choice_text = format_choices(choices)
    task = f"{subject}\n\n{question}\n\nChoices:\n{choice_text}"
    positive = (
        "Role: You are a meticulous logician focused on absolute precision.\n"
        "Task: Derive the answer to the following question using proof-level rigor.\n"
        "Instructions:\n"
        "• Derive: Do not just state facts; deduce them from axioms or given data.\n"
        "• Verify: Check each intermediate calculation or logic jump for errors.\n"
        "• Precision: Prioritize correctness over fluency. Reject any heuristic shortcuts.\n"
        "• Output: Provide a sound explanation.\n"
        f"Question:\n{task}\n"
        "Rigorous Derivation:"
    )
    negative = (
        "Role: You are a fluent conversationalist acting on \"autopilot\".\n"
        "Task: Provide a plausibly sounding answer based on surface-level associations.\n"
        "Instructions:\n"
        "• Flow: Write whatever comes to mind first based on language patterns.\n"
        "• Approximate: Do not perform actual calculations or verification. Use \"ballpark\" figures.\n"
        "• Plausibility: The answer should sound correct to a layperson, even if the logic is flawed.\n"
        "• Output: Generate a coherent but unverified response (simulate a hallucination if necessary).\n"
        f"Question:\n{task}\n"
        "Plausible Response:"
    )
    return task, positive, negative


def load_samples(
    dataset_name: str,
    subjects: list[str],
    split: str,
    num_samples: int,
    seed: int,
    cache_dir: str | None,
) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "This script requires Hugging Face Datasets. Install it with: "
            "python -m pip install datasets"
        ) from exc

    grouped: dict[str, list[dict]] = defaultdict(list)
    for subject in subjects:
        LOGGER.info("Loading MMLU subject %s (%s split)", subject, split)
        kwargs = {"split": split}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        dataset = load_dataset(dataset_name, subject, **kwargs)
        for index, item in enumerate(dataset):
            question = str(item["question"])
            choices = list(item["choices"])
            task, positive, negative = make_prompt_pair(subject, question, choices)
            grouped[subject].append(
                {
                    "id": f"mmlu_{subject}_{index}",
                    "task_id": f"mmlu_{subject}_{index}",
                    "subject": subject,
                    "question": question,
                    "choices": choices,
                    "answer": item.get("answer"),
                    "task": task,
                    "positive_prompt": positive,
                    "negative_prompt": negative,
                    "prompt_format": "chat_template:user+generation_prompt:contrastive_riser_prompts",
                }
            )

    if not grouped:
        raise RuntimeError("No MMLU examples were loaded")

    rng = random.Random(seed)
    for rows in grouped.values():
        rng.shuffle(rows)

    available = sum(len(rows) for rows in grouped.values())
    if available < num_samples:
        raise RuntimeError(
            f"Requested {num_samples} examples, but the selected MMLU subjects "
            f"contain only {available} examples in split {split!r}"
        )

    # Round-robin sampling keeps a large subject from dominating the vector set.
    selected = []
    cursors = {subject: 0 for subject in subjects if subject in grouped}
    while len(selected) < num_samples:
        for subject in subjects:
            rows = grouped.get(subject, [])
            cursor = cursors.get(subject, 0)
            if cursor >= len(rows):
                continue
            selected.append(rows[cursor])
            cursors[subject] = cursor + 1
            if len(selected) == num_samples:
                break

    LOGGER.info("Selected %d MMLU mathematics examples", len(selected))
    return selected


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model name/path")
    parser.add_argument("--dataset-name", default="cais/mmlu")
    parser.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir")
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--aggregation", choices=["last", "concat", "mean"], default="last")
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument(
        "--prompt-pairs-output",
        default="artifacts/mmlu_math_500_prompt_pairs.jsonl",
    )
    parser.add_argument(
        "--vectors-output",
        default="artifacts/mmlu_math_500_vectors.pt",
    )
    parser.add_argument(
        "--library-output",
        default="artifacts/mmlu_math_500_primitives.pt",
    )
    parser.add_argument(
        "--metadata-output",
        default="artifacts/mmlu_math_500_primitives.json",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.clusters <= 0:
        raise ValueError("--clusters must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    dtype = args.dtype
    if dtype is None:
        dtype = "float16" if args.device.startswith("cuda") else "float32"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    samples = load_samples(
        dataset_name=args.dataset_name,
        subjects=args.subjects,
        split=args.split,
        num_samples=args.num_samples,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    write_jsonl(samples, Path(args.prompt_pairs_output))

    extractor = ActivationExtractor(
        model_name=args.model,
        target_layers=args.layers,
        device=args.device,
        dtype=dtype,
    )
    prompt_pairs = [
        (
            row["positive_prompt"],
            row["negative_prompt"],
            row["task"],
            row["task_id"],
        )
        for row in samples
    ]
    activation_pairs = extractor.extract_batch(
        prompt_pairs,
        max_length=args.max_length,
        layer_aggregation=args.aggregation,
    )

    positive = torch.stack([pair.positive_activation for pair in activation_pairs])
    negative = torch.stack([pair.negative_activation for pair in activation_pairs])
    differences = torch.stack([pair.difference for pair in activation_pairs])
    vectors_path = Path(args.vectors_output)
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "positive_activations": positive,
            "negative_activations": negative,
            "differences": differences,
            "task_ids": [pair.task_id for pair in activation_pairs],
            "tasks": [pair.task for pair in activation_pairs],
            "model": args.model,
            "target_layers": args.layers,
            "layer_aggregation": args.aggregation,
            "num_samples": len(activation_pairs),
            "prompt_format": "chat_template:user+generation_prompt:contrastive_riser_prompts",
        },
        vectors_path,
    )

    n_clusters = min(args.clusters, len(activation_pairs))
    clustering = PrimitiveClustering(n_clusters=n_clusters)
    primitives = clustering.fit(activation_pairs)
    primitive_names = [f"primitive_{index}" for index in range(n_clusters)]
    for primitive_id in primitives:
        primitive_names[primitive_id] = f"mmlu_math_cluster_{primitive_id}"
    library = PrimitiveLibrary(
        primitives=primitives,
        primitive_names=primitive_names,
        metadata={
            "source": "MMLU mathematics subjects",
            "dataset_name": args.dataset_name,
            "split": args.split,
            "subjects": args.subjects,
            "model": args.model,
            "target_layers": args.layers,
            "layer_aggregation": args.aggregation,
            "num_pairs": len(activation_pairs),
            "positive_behavior": "proof-level rigorous reasoning",
            "negative_behavior": "plausible but unverified response",
            "prompt_format": "chat_template:user+generation_prompt:contrastive_riser_prompts",
        },
    )
    library.save(args.library_output, metadata_path=args.metadata_output)

    print(f"Saved prompt pairs to {args.prompt_pairs_output}")
    print(f"Saved raw activation vectors to {args.vectors_output}")
    print(f"Saved {library.num_primitives} primitives to {args.library_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
