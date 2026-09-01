from pathlib import Path

import pytest

from self_steering.config import ConfigError, load_config


def test_load_config_deep_merges_three_yaml_files(tmp_path: Path) -> None:
    (tmp_path / "model.yaml").write_text(
        "model:\n  name: qwen\n  dtype: bfloat16\n  num_hidden_layers: 28\n",
        encoding="utf-8",
    )
    (tmp_path / "data.yaml").write_text(
        "data:\n  enabled_steering_datasets: [math500]\n",
        encoding="utf-8",
    )
    (tmp_path / "experiment.yaml").write_text(
        "experiment:\n  target_layer: 19\n  high_demand_threshold: 4\n"
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
        "model:\n  num_hidden_layers: 28\n"
        "data:\n  enabled_steering_datasets: [math500]\n"
        "experiment:\n  target_layer: 19\n  high_demand_threshold: 4\n"
        "  low_demand_threshold: 1\n  vector_scaling: mean_norm\n"
        "  alphas: [-1, 0, 1]\n",
        encoding="utf-8",
    )

    config = load_config(
        [path],
        overrides=["experiment.target_layer=18", "data.enabled_steering_datasets=[math500, aime2026]"],
    )

    assert config["experiment"]["target_layer"] == 18
    assert config["data"]["enabled_steering_datasets"] == ["math500", "aime2026"]


def test_load_config_rejects_out_of_range_layer(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "model:\n  num_hidden_layers: 28\n"
        "data:\n  enabled_steering_datasets: [math500]\n"
        "experiment:\n  target_layer: 28\n  high_demand_threshold: 4\n"
        "  low_demand_threshold: 1\n  vector_scaling: mean_norm\n"
        "  alphas: [-1, 0, 1]\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="target_layer"):
        load_config([path])


def test_load_config_rejects_unknown_dataset(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "model:\n  num_hidden_layers: 28\n"
        "data:\n  enabled_steering_datasets: [unknown]\n"
        "experiment:\n  target_layer: 19\n  high_demand_threshold: 4\n"
        "  low_demand_threshold: 1\n  vector_scaling: mean_norm\n"
        "  alphas: [-1, 0, 1]\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="unknown dataset"):
        load_config([path])

