> **Historical prototype:** this repo is kept public for development history, but it is not the current first-contact path for the TS research stack. Start with `TS-Start-Here`, `TS-Reasoner-v0`, `TS-Codex-OS`, `TS-Core`, `bozo` / TensionLM, and `BoggersTheCIG`.

# schizo_bet

Small-language-model experiments built around a **TS-scale** backbone: stacked causal **tension-graph** layers (learned pairwise gates on a local past window), oscillatory modulation, periodic MoE feed-forward blocks, and mid-stack Schrödinger-style VQ. Training is file-driven with optional byte or BPE tokenization, AMP on CUDA, learning-rate warmup/decay, and checkpointing.

## Requirements

Python 3.10+ recommended.

```bash
pip install -r requirements-scale.txt
```

## Train

From the repository root:

```bash
python -m ts_scale.train --train-file data/train.txt --eval-file data/eval.txt --cuda --amp
```

See `python -m ts_scale.train --help` for sequence length, model size, tokenizer choice, checkpoint paths, and resume options.

## Layout

| Path | Role |
|------|------|
| `ts_scale/` | Model, losses, training loop, tokenizers, checkpoint IO |
| `ts_llm/` | Companion LM/embeddings utilities (`python -m ts_llm`) |
| `data/` | Example train/eval text (replace with your corpus) |
| `requirements-scale.txt` | PyTorch, NumPy, tokenizers |

Training checkpoints are not committed (see `.gitignore`); use `--checkpoint-dir` to write them locally.

## License

This project is licensed under the [MIT License](LICENSE).
