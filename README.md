# TinyML

CIFAR-10 classifier for a microcontroller budget (<= 100 KB flash, <= 20 KB peak activation RAM, int8).

Pipeline (`tinyml/pipeline.py`), each stage distilled from the teacher:

1. **Teacher**: ResNet-18, fp32.
2. **Once-for-All supernet** (`tinyml/ofa.py`): MobileNetV2-style, elastic width/depth/kernel/expand,
   trained in fp32 with the sandwich rule (largest + smallest + 2 random subnets per batch), KD from the teacher, early stopping.
3. **Search** (`tinyml/search.py`): evolutionary search under the flash/RAM budget, val accuracy after BN
   recalibration, early stop when the best stalls.
4. **Structured pruning** (`tinyml/prune.py`) of hidden/head channels, then KD fine-tune.
5. **QAT** (`tinyml/qat.py`) last, with KD: int8 per-channel weights, per-tensor activations, BN folded.

Baselines kept for comparison: the searched arch trained from scratch (CE and KD), the subnet before
fine-tuning, pruning without fine-tune and with CE-only fine-tune, and PTQ before QAT.
Results land in `runs/<name>/results.md`.

## Usage

```bash
kaggle datasets download pankrzysiu/cifar10-python -p data/raw --unzip
uv sync
uv run python -m tinyml.pipeline --data data/raw --run-dir runs/main
```

Every stage resumes from its checkpoint on re-run. `--smoke` runs the full pipeline on tiny data in seconds.

## Branches

- `prod`: production-ready code
- `dev`: integration branch for ongoing work
- `test`: testing branch

## Layout

- `tinyml/`: pipeline code
- `test/`: unit, smoke and integration tests (`uv run pytest`); integration tests skip without CUDA or the real dataset
- `.github/workflows/ci.yml`: CI pipeline, runs on pushes and PRs to `prod`, `dev`, `test`
