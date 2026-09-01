from _common import build_parser, print_paths, resolved_config


def main() -> None:
    parser = build_parser("Run continuous generation-time capability steering")
    args = parser.parse_args()
    config = resolved_config(args)
    from self_steering.models.loader import load_model_and_tokenizer
    from self_steering.pipeline import run_steering

    model, tokenizer = load_model_and_tokenizer(config["model"])
    print_paths(run_steering(config, model, tokenizer))


if __name__ == "__main__":
    main()

