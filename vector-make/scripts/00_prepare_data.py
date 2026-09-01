from _common import build_parser, print_paths, resolved_config


def main() -> None:
    parser = build_parser("Prepare canonical MMLU and steering datasets")
    args = parser.parse_args()
    from self_steering.datasets.registry import DatasetRegistry
    from self_steering.pipeline import prepare_data

    print_paths(prepare_data(resolved_config(args), DatasetRegistry.default()))


if __name__ == "__main__":
    main()

