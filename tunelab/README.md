# tunelab

> Post-training experiments. Train a small open model to do what Sonnet 4.6 does on a specific task — 100× cheaper.

**🔬 Reproducibility-first repo for one RL-post-training experiment.** Closes the RL gap in the portfolio: this is where the verifiable-reward / GRPO-style post-training story lives.

## The experiment

**Task:** *Structured prior-authorization extraction.* Input is a free-text patient summary; output is a Pydantic-validating JSON object capturing diagnosis, medications tried, contraindications, and red flags. This is the kind of structured extraction that powers PriorAuth's drafter — currently done by Claude Sonnet 4.6. The thesis: a 1B-parameter open model trained with **verifiable rewards (GRPO)** on synthetic + real examples can do it well enough for ~$0.00005 per call instead of ~$0.003.

**Reward function** (`src/tunelab/reward.py`):

1. *Schema-validity reward* (binary). Does the output parse as JSON and validate against the `PriorAuthExtraction` Pydantic schema? Either 1.0 or 0.0.
2. *Field-match reward* (graded). For each expected field, did the model produce the right value? Exact match for enums / IDs, fuzzy match for free-text.
3. *Combined reward.* `0.3 * schema_ok + 0.7 * field_match`. Schema-validity is gated — invalid JSON gets full zero.

**Eval harness** (`src/tunelab/eval.py`) — feeds the eval set through any callable `model_fn: str → str` and computes reward stats. Lets you compare:
- Baseline (raw 1B model, zero-shot)
- Few-shot prompted baseline
- Post-trained checkpoint

**Training** (`src/tunelab/train.py`) — uses TRL's GRPOTrainer on a small base model (Qwen 2.5 1.5B or Llama 3.2 1B) with LoRA adapters. Designed to run on a single T4 / A10 GPU in Colab.

## Why this is the right RL portfolio piece

- **Verifiable rewards.** No human in the loop, no preference data — the reward function is deterministic Python. This is the post-DeepSeek-R1 / RLVR / GRPO direction.
- **Connects to existing portfolio work.** The output schema matches what PriorAuth needs from its drafter. The trained model could plug into PriorAuth as a cheap alternative to Sonnet.
- **Honest about cost/quality tradeoffs.** The eval harness measures the actual gap. The pitch isn't "I replaced Sonnet" — it's "I quantified what a fine-tuned 1B can do on this task."

## Layout

```
tunelab/
├── src/tunelab/
│   ├── schema.py        ← Pydantic output: PriorAuthExtraction
│   ├── reward.py        ← schema_validity + field_match rewards
│   ├── dataset.py       ← synthetic note-and-expected-JSON pairs
│   ├── eval.py          ← run a model_fn against the eval set, get a report
│   ├── train.py         ← GRPO training script (heavy deps, opt-in)
│   └── cli.py
├── data/
│   ├── train.jsonl      ← 100 synthetic examples for training
│   └── eval.jsonl       ← 20 held-out examples for evaluation
├── notebooks/
│   └── train.ipynb      ← Colab-friendly notebook (you run this on GPU)
└── tests/               ← reward + schema tests, no torch needed
```

## Quick start — explore without training

```bash
cd tunelab
uv sync
uv run pytest tests/         # reward + schema tests pass without torch
uv run tunelab inspect-dataset
uv run tunelab eval-mock     # run the eval harness against a fake model
```

## Quick start — actually train

```bash
cd tunelab
uv sync --extra train        # installs torch, trl, transformers — heavy
# Either run locally on a GPU box:
uv run tunelab train --base-model Qwen/Qwen2.5-1.5B --output-dir ./runs/v1
# Or open notebooks/train.ipynb in Colab and run there with a T4.
```

## Roadmap

- **v0.1** *(current)* — Schema + reward + 100-example synthetic dataset + eval harness + GRPO training script (untrained, ready to run).
- **v0.2** — Run the training, publish before/after eval numbers + the LoRA checkpoint to Hugging Face.
- **v0.3** — Swap PriorAuth's drafter to optionally use the fine-tuned model. Cost/quality comparison in the PriorAuth UI.

## License

MIT.
