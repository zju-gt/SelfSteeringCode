from _common import build_parser, print_paths, resolved_config


def main() -> None:
    parser = build_parser("Aggregate activation contrasts into capability vectors")
    args = parser.parse_args()
    from self_steering.pipeline import extract_vectors

    print_paths(extract_vectors(resolved_config(args)))


if __name__ == "__main__":
    main()

