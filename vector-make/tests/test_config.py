from pathlib import Path

import pytest

from self_steering.config import ConfigError, load_config, validate_config


CAPABILITIES_YAML = "  capabilities: [QLl, QLq, CL, MCr]\n"
PINNED_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def valid_config() -> dict:
    return {
        "model": {
            "num_hidden_layers": 28,
            "max_new_tokens": 16,
            "revision": PINNED_REVISION,
        },
        "data": {"enabled_steering_datasets": ["math500"]},
        "experiment": {
            "capabilities": ["QLl", "QLq", "CL", "MCr"],
            "target_layer": 19,
            "high_demand_threshold": 4,
            "low_demand_threshold": 1,
            "vector_scaling": "mean_norm",
            "alphas": [-1, 0, 1],
            "annotation": {
                "max_workers": 2,
                "max_attempts": 3,
                "initial_backoff_seconds": 0.1,
            },
        },
    }


def test_load_config_deep_merges_three_yaml_files(tmp_path: Path) -> None:
    (tmp_path / "model.yaml").write_text(
        "model:\n  name: qwen\n  dtype: bfloat16\n  num_hidden_layers: 28\n"
        f"  revision: {PINNED_REVISION}\n",
        encoding="utf-8",
    )
    (tmp_path / "data.yaml").write_text(
        "data:\n  enabled_steering_datasets: [math500]\n",
        encoding="utf-8",
    )
    (tmp_path / "experiment.yaml").write_text(
        "experiment:\n" + CAPABILITIES_YAML + "  target_layer: 19\n  high_demand_threshold: 4\n"
        "  low_demand_threshold: 1\n  vector_scaling: mean_norm\n"
        "  alphas: [-1, 0, 1]\n",
        encoding="utf-8",
    )

    config = load_config(
        [
            tmp_path / "model.yaml",
            tmp_path / "data.yaml",
            tmp_path / "experiment.yaml",
        ]
    )

    assert config["model"]["name"] == "qwen"
    assert config["data"]["enabled_steering_datasets"] == ["math500"]
    assert config["experiment"]["target_layer"] == 19


def test_load_config_applies_dot_path_override(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"model:\n  num_hidden_layers: 28\n  revision: {PINNED_REVISION}\n"
        "data:\n  enabled_steering_datasets: [math500]\n"
        "experiment:\n" + CAPABILITIES_YAML + "  target_layer: 19\n  high_demand_threshold: 4\n"
        "  low_demand_threshold: 1\n  vector_scaling: mean_norm\n"
        "  alphas: [-1, 0, 1]\n",
        encoding="utf-8",
    )

    config = load_config(
        [path],
        overrides=[
            "experiment.target_layer=18",
            "data.enabled_steering_datasets=[math500, aime2026]",
        ],
    )

    assert config["experiment"]["target_layer"] == 18
    assert config["data"]["enabled_steering_datasets"] == ["math500", "aime2026"]


def test_load_config_rejects_out_of_range_layer(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        f"model:\n  num_hidden_layers: 28\n  revision: {PINNED_REVISION}\n"
        "data:\n  enabled_steering_datasets: [math500]\n"
        "experiment:\n" + CAPABILITIES_YAML + "  target_layer: 28\n  high_demand_threshold: 4\n"
        "  low_demand_threshold: 1\n  vector_scaling: mean_norm\n"
        "  alphas: [-1, 0, 1]\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="target_layer"):
        load_config([path])


def test_load_config_rejects_unknown_dataset(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        f"model:\n  num_hidden_layers: 28\n  revision: {PINNED_REVISION}\n"
        "data:\n  enabled_steering_datasets: [unknown]\n"
        "experiment:\n" + CAPABILITIES_YAML + "  target_layer: 19\n  high_demand_threshold: 4\n"
        "  low_demand_threshold: 1\n  vector_scaling: mean_norm\n"
        "  alphas: [-1, 0, 1]\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="unknown dataset"):
        load_config([path])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda config: config["experiment"].update(capabilities=["QLl"]), "capabilities"),
        (lambda config: config["experiment"].update(alphas=[0, 1, 1]), "unique"),
        (lambda config: config["experiment"].update(alphas=[0, float("inf")]), "finite"),
        (lambda config: config["model"].update(max_new_tokens=0), "max_new_tokens"),
        (lambda config: config["model"].update(revision="main"), "revision"),
        (
            lambda config: config["experiment"]["annotation"].update(max_workers=0),
            "max_workers",
        ),
        (
            lambda config: config["experiment"]["annotation"].update(max_attempts=0),
            "max_attempts",
        ),
        (
            lambda config: config["experiment"]["annotation"].update(
                initial_backoff_seconds=0
            ),
            "initial_backoff_seconds",
        ),
    ],
)
def test_validate_config_rejects_invalid_mvp_parameters(mutation, message) -> None:
    config = valid_config()
    mutation(config)
    with pytest.raises(ConfigError, match=message):
        validate_config(config)
