from _common import build_parser, print_paths, resolved_config


def main() -> None:
    parser = build_parser("Compute capability-vector coherence and cosine similarity")
    args = parser.parse_args()
    from self_steering.pipeline import analyze_similarity

    print_paths(analyze_similarity(resolved_config(args)))


if __name__ == "__main__":
    main()

