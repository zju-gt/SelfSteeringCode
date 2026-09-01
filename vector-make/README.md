# Self-Steering MVP

This project tests whether four DeLeAn cognitive capabilities correspond to stable and causally steerable residual-stream directions in `Qwen/Qwen2.5-7B-Instruct`.

## Experiment defaults

- Extraction dataset: MMLU.
- Steering dataset: MATH500.
- Optional steering datasets: AIME 2024, AIME 2025, AIME 2026, ARC-Challenge, and OpenBookQA.
- Capabilities: `QLl`, `QLq`, `CL`, and `MCr`.
- Decoder layer: zero-based index 19.
- Extraction filter: `d_k >= 4`.
- Low-demand control: `d_k <= 1`.
- Steering coefficients: `-1`, `-0.5`, `0`, `0.5`, and `1`.
- Decoding: greedy CoT generation.

## Server installation

Use a clean Python 3.10 or newer environment. Choose a PyTorch build appropriate for the server CUDA driver before installing this project; the project intentionally does not pin workstation-specific CUDA wheels.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Set the annotation credential before stage 01:

```bash
export OPENAI_API_KEY="your-key"
```

Hugging Face authentication is optional for public sources. Set `HF_TOKEN` if the server cache or mirror requires it.

## Configuration

The three YAML files are merged left-to-right:

```text
configs/model.yaml
configs/data.yaml
configs/experiment.yaml
```

Any value can be changed without editing code:

```bash
python scripts/06_run_steering.py \
  --config configs/model.yaml \
  --config configs/data.yaml \
  --config configs/experiment.yaml \
  --override experiment.target_layer=18 \
  --override data.enabled_steering_datasets=[math500,aime2024,aime2025,aime2026]
```

Local JSONL sources override Hugging Face downloads:

```bash
--override data.sources.aime2026.local_path=/data/aime2026.jsonl
```

## Pipeline

Run all commands from the `vector-make` directory:

```bash
python scripts/00_prepare_data.py --config configs/model.yaml --config configs/data.yaml --config configs/experiment.yaml
python scripts/01_score_demands.py --config configs/model.yaml --config configs/data.yaml --config configs/experiment.yaml
python scripts/02_prepare_items.py --config configs/model.yaml --config configs/data.yaml --config configs/experiment.yaml
python scripts/03_capture_contrasts.py --config configs/model.yaml --config configs/data.yaml --config configs/experiment.yaml
python scripts/04_extract_vectors.py --config configs/model.yaml --config configs/data.yaml --config configs/experiment.yaml
python scripts/05_analyze_similarity.py --config configs/model.yaml --config configs/data.yaml --config configs/experiment.yaml
python scripts/06_run_steering.py --config configs/model.yaml --config configs/data.yaml --config configs/experiment.yaml
python scripts/07_score_generations.py --config configs/model.yaml --config configs/data.yaml --config configs/experiment.yaml
```

Stages use incremental JSONL or per-item safetensors artifacts. Re-running a stage resumes completed annotation, capture, or generation keys.

Use `--limit N` on stages 00, 01, 03, and 06 for a small end-to-end server smoke run. The limit is applied per dataset (and per capability during capture), including when a preceding stage prepared a larger file.

## Outputs

```text
data/processed/             canonical data and demand slices
data/scored/                long and wide DeLeAn annotations
outputs/activations/<id>/   indexed per-item capability contrast shards
outputs/vectors/<id>/       raw, unit, and mean-norm vectors
outputs/generations/<id>.jsonl
                            resumable steered generations
outputs/metrics/<id>_*      similarity, coherence, paired accuracy, and specificity
outputs/manifests/          resolved run metadata
```

## Tests

The default test suite is offline and does not download Qwen or call OpenAI:

```bash
python -m pytest -q
python -m compileall -q src scripts tests
```

Passing the offline suite verifies pipeline logic only. It does not claim that a full 7B GPU run or live annotation job has completed.
