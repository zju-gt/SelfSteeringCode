from pathlib import Path

from _common import build_parser, print_paths, resolved_config


def main() -> None:
    parser = build_parser("Score four DeLeAn demand dimensions")
    args = parser.parse_args()
    config = resolved_config(args)

    from openai import OpenAI

    from self_steering.datasets.delean_labeler import (
        AnnotationRequest,
        expected_annotation_key,
        label_one_dimension,
        retry_call,
    )
    from self_steering.pipeline import score_demands

    annotation = config["experiment"]["annotation"]
    model = str(annotation["model"])
    rubric_dir = Path(config["experiment"]["paths"].get("rubrics_dir", "rubrics"))
    client = OpenAI()
    rubrics = {
        dimension: (rubric_dir / f"{dimension}.txt").read_text(encoding="utf-8")
        for dimension in config["experiment"]["capabilities"]
    }

    def request_for(item, dimension):
        return AnnotationRequest(
            item_id=item.item_id,
            dataset=item.dataset,
            split=item.split,
            prompt=item.prompt,
            dimension=dimension,
            domain=str(item.metadata.get("subject"))
            if item.metadata.get("subject")
            else None,
        )

    def label(item, dimension):
        request = request_for(item, dimension)
        return retry_call(
            lambda: label_one_dimension(
                client,
                request,
                rubric_dir / f"{dimension}.txt",
                model=model,
            ),
            max_attempts=int(annotation.get("max_attempts", 5)),
            initial_backoff_seconds=float(
                annotation.get("initial_backoff_seconds", 1.0)
            ),
        )

    def key(item, dimension):
        return expected_annotation_key(
            request_for(item, dimension), rubrics[dimension], model
        )

    print_paths(score_demands(config, label, expected_key_fn=key))


if __name__ == "__main__":
    main()
