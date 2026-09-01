from _common import build_parser, print_paths, resolved_config


def main() -> None:
    parser = build_parser("Score generations and build causal specificity matrices")
    args = parser.parse_args()
    from self_steering.pipeline import score_generations

    print_paths(score_generations(resolved_config(args)))


if __name__ == "__main__":
    main()

