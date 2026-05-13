"""GRPO post-training script for the structured-extraction task.

REQUIRES the `--extra train` deps: torch, transformers, trl, accelerate, peft.
NOT auto-runnable in CI — by design. Run this on a GPU (Colab T4, A10, etc.).

Architecture:
- Base model: small open chat model (Qwen 2.5 1.5B by default).
- Adapter: LoRA via PEFT (so we don't fine-tune the whole model — much
  smaller checkpoint, faster training, no full-precision GPU memory needed).
- Reward: combined_reward from tunelab.reward (schema-validity + field-match).
- Trainer: TRL's GRPOTrainer.

Usage:
    uv run tunelab train --base-model Qwen/Qwen2.5-1.5B --output-dir runs/v1

Or open notebooks/train.ipynb in Colab on a T4.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tunelab.dataset import load_train
from tunelab.reward import combined_reward


SYSTEM_PROMPT = """You extract structured prior-authorization data from a free-text patient summary.

Output ONLY a JSON object that validates against this schema:
{
  "primary_diagnosis": str,
  "diagnosis_duration_years": float | null,
  "medications": [{"name": str, "outcome": one of [tried_worked, tried_intolerance, tried_ineffective, tried_hypoglycemia, not_tried], "note": str}],
  "contraindications": [str],
  "red_flags": [str],
  "requested_service": str
}

No prose, no code fences, no explanation. JSON only."""


def reward_fn(prompts: list[str], completions: list[str], expecteds: list[dict[str, Any]], **kwargs: Any) -> list[float]:
    """The GRPO trainer calls this for each generated completion.

    Returns a list of scalar rewards, one per completion. We use the
    `combined_reward` (schema-validity + field-match) defined in reward.py.
    """
    return [combined_reward(c, e)["reward"] for c, e in zip(completions, expecteds)]


def main(
    base_model: str = "Qwen/Qwen2.5-1.5B",
    output_dir: str = "runs/v1",
    num_train_epochs: int = 3,
    learning_rate: float = 5e-6,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
) -> None:
    """Run GRPO training.

    Imports the heavy deps (torch, trl, transformers) inside main() so this
    file can be imported in CI without dragging them in.
    """
    try:
        import torch  # noqa: F401
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        raise RuntimeError(
            "Training dependencies not installed. Run `uv sync --extra train` "
            "on a GPU box, then retry. Error was: " + str(e)
        ) from e

    # ---- dataset ----
    examples = load_train()
    ds = Dataset.from_list([
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex.input},
            ],
            "expected": ex.expected,
        }
        for ex in examples
    ])

    # ---- model + tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto")

    # ---- LoRA ----
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # ---- GRPO config ----
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=4,            # GRPO compares N samples per prompt
        max_completion_length=512,
        save_strategy="epoch",
        logging_steps=10,
        report_to="none",             # set to "wandb" if you want W&B logging
    )

    # GRPO wants a single reward function; we wrap to inject `expecteds` from the dataset.
    def _reward_wrapper(prompts: list[str], completions: list[str], **kwargs: Any) -> list[float]:
        # GRPOTrainer passes the dataset row through kwargs; the row's "expected" is per-prompt.
        expecteds = kwargs.get("expected") or []
        if not isinstance(expecteds, list):
            expecteds = [expecteds] * len(completions)
        return reward_fn(prompts, completions, expecteds)

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=ds,
        processing_class=tokenizer,
        reward_funcs=[_reward_wrapper],
    )
    trainer.train()
    trainer.save_model(str(Path(output_dir) / "final"))
    print(f"Saved LoRA adapter to {Path(output_dir) / 'final'}")
