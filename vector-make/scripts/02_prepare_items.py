from _common import build_parser, print_paths, resolved_config


def main() -> None:
    parser = build_parser("Build high-demand extraction and high/low evaluation slices")
    args = parser.parse_args()
    from self_steering.pipeline import prepare_items

    print_paths(prepare_items(resolved_config(args)))


if __name__ == "__main__":
    main()
