from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOTrainer, GRPOConfig
import torch
from src.config import Config, EvalConfig
from src.rewards.bioinformatics.scorer import Scorer
from src.rewards.bioinformatics.logger import RewardComponentLogger
from src.eval.eval import Evaluator
from src.utils.training_utils import EvalCallback, test_checkpoint_directory_write
from src.ablations import get_ablation_config
from vllm import SamplingParams
import datetime
from typing import List
import wandb
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import re
import os


def run_grpo(ablation_name: str = "full_reward"):
    """Run GRPO training with the specified ablation configuration.

    Args:
        ablation_name: Name of the ablation config (see src/ablations.py).
    """
    cfg = Config()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"grpo-{ablation_name}-{timestamp}"

    # Dataset loading
    def load_train_val_datasets():
        """Load and preprocess training and validation datasets."""
        def select_prompt_column(ds):
            cols = set(ds.column_names)
            keep_cols = ["prompt"] + [c for c in ["data_source", "ability", "reward_model", "extra_info"] if c in cols]
            return ds.select_columns(keep_cols)

        train_ds = load_dataset("parquet", data_files=cfg.train_dataset, split="train")
        val_ds = load_dataset("parquet", data_files=cfg.val_dataset, split="train")

        return select_prompt_column(train_ds), select_prompt_column(val_ds)

    train_ds, eval_ds = load_train_val_datasets()

    # Tokenizer setup
    tok = AutoTokenizer.from_pretrained(cfg.model, use_fast=True, trust_remote_code=True)
    tok.padding_side = "left"
    tok.eos_token = "</s>"
    tok.bos_token = "<s>"
    tok.pad_token = "[PAD]"

    # Validate token IDs
    assert tok.eos_token_id == 30001, f"Expected eos_token_id=30001, got {tok.eos_token_id}"
    assert tok.bos_token_id == 30000, f"Expected bos_token_id=30000, got {tok.bos_token_id}"
    assert tok.pad_token_id == 3, f"Expected pad_token_id=3, got {tok.pad_token_id}"

    # Model initialization kwargs
    model_init_kwargs = {
        "trust_remote_code": True,
        "eos_token_id": tok.eos_token_id,
        "bos_token_id": tok.bos_token_id,
        "pad_token_id": tok.pad_token_id,
    }

    # Training configuration - use /s3 mount point with prefix path
    checkpoint_dir = f"/s3/{cfg.checkpoints_path.rstrip('/')}/grpo-{ablation_name}/{run_name}"

    # Test checkpoint directory write access before proceeding
    test_checkpoint_directory_write(checkpoint_dir)

    args = GRPOConfig(
        model_init_kwargs=model_init_kwargs,
        output_dir=checkpoint_dir,

        # Training parameters
        num_train_epochs=20,
        learning_rate=cfg.grpo_learning_rate,
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        per_device_train_batch_size=cfg.grpo_per_device_train_batch_size,
        gradient_accumulation_steps=1,
        max_steps=-1,
        max_grad_norm=0.5,
        seed=42,

        # Logging and checkpointing
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,  # Keep last 5 checkpoints
        logging_strategy="steps",
        logging_steps=1,
        report_to=["wandb"],

        # Evaluation
        do_eval=True,
        eval_strategy="steps",
        eval_steps=50,  # Evaluate every 50 steps

        # Optimization
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=False,

        # GRPO-specific
        beta=cfg.grpo_beta,
        epsilon=cfg.grpo_epsilon,
        loss_type="bnpo",
        scale_rewards=True,
        mask_truncated_completions=False,
        disable_dropout=True,

        # Generation parameters
        remove_unused_columns=False,
        max_prompt_length=1024,
        num_generations=cfg.grpo_num_generations,
        max_completion_length=256,
        temperature=cfg.grpo_temperature,
        top_p=cfg.grpo_top_p,

        # vLLM configuration
        use_vllm=True,
        vllm_gpu_memory_utilization=0.15,
        vllm_mode="colocate",
    )

    # Reward configuration from ablation registry
    reward_config = get_ablation_config(ablation_name)

    # Initialize scorer and logger
    scorer = Scorer(reward_config)
    reward_logger = RewardComponentLogger(log_frequency=1)
    component_lock = Lock()

    # Initialize evaluation callback
    eval_config = EvalConfig(
        model_name=cfg.model,
        model_path=cfg.model,
        prompts_path=cfg.val_dataset,
        prompts_column="prompt",
        num_samples_per_prompt=5,
        overlap_merge_threshold=0.8,
        sampling_params=SamplingParams(
            max_tokens=256,
            temperature=0.95,
            top_p=0.90,
            top_k=0,
        ),
        write_to_wandb=True,
        wandb_project="plasmid-rl-icml-revision",
        wandb_run_name=run_name,
    )
    evaluator = Evaluator(eval_config)
    eval_callback = EvalCallback(evaluator)

    # HF model repo for this ablation
    hf_repo = f"McClain/plasmidgpt-rl-{ablation_name}"

    # Callback to push checkpoints to HuggingFace
    class HFPushCallback(TrainerCallback):
        """Push model checkpoints to HuggingFace Hub on save."""

        def on_save(self, args, state, control, **kwargs):
            step = state.global_step
            try:
                model = kwargs.get("model")
                if model is not None and hasattr(model, "push_to_hub"):
                    model.push_to_hub(
                        hf_repo,
                        revision=f"step-{step}",
                        commit_message=f"Checkpoint at step {step}",
                        private=True,
                    )
                    print(f"[HFPush] Pushed checkpoint step {step} to {hf_repo}")
            except Exception as e:
                print(f"[HFPush] Warning: Failed to push step {step}: {e}")

    # Sample table logging callback
    class SampleTableCallback(TrainerCallback):
        """Log sample sequences to W&B every 250 steps."""

        def __init__(self):
            self._step_samples: list = []

        def record_samples(self, prompts: List[str], completions: List[str], rewards: List[float]):
            """Record samples from the current batch for potential logging."""
            self._step_samples = list(zip(prompts, completions, rewards))

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 250 != 0 or not self._step_samples:
                return
            # Take up to 10 samples
            samples = self._step_samples[:10]
            table = wandb.Table(
                columns=["prompt_prefix", "completion", "reward", "length_bp"],
                data=[
                    [p[:50], c[:500], round(r, 4), len(c)]
                    for p, c, r in samples
                ],
            )
            try:
                wandb.log({"samples/examples": table}, step=state.global_step)
            except Exception as e:
                print(f"[SampleTable] Warning: Failed to log: {e}")
            self._step_samples = []

    sample_callback = SampleTableCallback()
    hf_push_callback = HFPushCallback()

    # Reward function
    def score_single(idx_and_seq):
        """Score a single sequence and log components thread-safely."""
        idx, seq = idx_and_seq
        try:
            score, components = scorer.score(seq)
            with component_lock:
                reward_logger.add_components(components, float(score))
            return float(score), components
        except Exception as e:
            print(f"Warning: Failed to score completion {idx} (len={len(seq)}): {str(e)[:100]}")
            return 0.0, None

    def batch_reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        """Compute rewards for a batch of completions."""
        # Clean sequences: remove non-DNA characters
        cleaned = [re.sub(r'[^ATCG]', '', c.upper().replace(" ", "")) for c in completions]

        # Parallelize scoring
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(score_single, enumerate(cleaned)))

        rewards = [r[0] for r in results]

        # Record samples for table logging
        sample_callback.record_samples(prompts, cleaned, rewards)

        return rewards

    # Data locations
    s3_base = f"s3://phd-research-storage-1758274488/icml-revision/ablations/{ablation_name}"

    # Describe what this ablation disables
    _ablation_descriptions = {
        "full_reward": "Control run: all reward components enabled (identical to production config).",
        "no_repeat_penalty": "Repeat penalty disabled (repeat_penalty_enabled=False). All other components active.",
        "no_length_prior": "Length prior disabled (length_reward_mode=False). No reward/penalty based on sequence length.",
        "no_cassette_bonus": "Cassette arrangement bonus disabled (location_aware=False). CDS detection still active via Prodigal.",
        "cds_only": "Only CDS scoring active. All other component weights zeroed. No length, repeat, or cassette rewards.",
        "length_only": "Only length prior active. All annotation-based scoring at epsilon weight (0.001).",
    }

    # Initialize W&B
    wandb_run = wandb.init(
        project="plasmid-rl-icml-revision",
        entity=cfg.wandb_entity,
        name=run_name,
        group="ablation-training",
        tags=["icml-revision", ablation_name],
        notes=(
            f"ICML revision ablation: {ablation_name}\n\n"
            f"{_ablation_descriptions.get(ablation_name, '')}\n\n"
            f"## Data Locations\n"
            f"- **HuggingFace model**: https://huggingface.co/{hf_repo}\n"
            f"- **S3 checkpoints**: {s3_base}/checkpoints/\n"
            f"- **S3 wandb logs**: {s3_base}/wandb/\n"
            f"- **Base model**: {cfg.model}\n"
            f"- **Training data**: {cfg.train_dataset}\n"
        ),
        config={
            "ablation_config": ablation_name,
            "model": cfg.model,
            "reward_config": reward_config.model_dump(),
            "training": {
                "learning_rate": cfg.grpo_learning_rate,
                "batch_size": cfg.grpo_per_device_train_batch_size,
                "num_epochs": args.num_train_epochs,
                "num_generations": cfg.grpo_num_generations,
            },
            "grpo": {
                "beta": cfg.grpo_beta,
                "epsilon": cfg.grpo_epsilon,
                "temperature": cfg.grpo_temperature,
                "top_p": cfg.grpo_top_p,
                "loss_type": args.loss_type,
            },
            "checkpoint_dir": checkpoint_dir,
            "hf_repo": hf_repo,
            "s3_checkpoints": f"{s3_base}/checkpoints/",
            "anyscale_job_id": os.environ.get("ANYSCALE_JOB_ID", "local"),
        },
    )

    # Print wandb URL and checkpoint info
    if wandb_run:
        print(f"\n{'='*80}")
        print(f"W&B Run URL: {wandb_run.url}")
        print(f"Checkpoint Directory: {checkpoint_dir}")
        print(f"Ablation Config: {ablation_name}")
        print(f"HF Repo: {hf_repo}")
        print(f"{'='*80}\n")

    # Initialize trainer
    trainer = GRPOTrainer(
        model=cfg.model,
        reward_funcs=[batch_reward_fn],
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        callbacks=[reward_logger, eval_callback, sample_callback, hf_push_callback],
    )

    # Set trainer reference in callback (for accessing trainer.llm)
    eval_callback.set_trainer(trainer)

    # Train and save
    print(f"Starting training with {args.num_train_epochs} epochs...")
    trainer.train()

    # Save final model and tokenizer
    print(f"Saving final model to {checkpoint_dir}...")
    trainer.save_model(checkpoint_dir)
    tok.save_pretrained(checkpoint_dir)

    # Push final model to HuggingFace
    try:
        trainer.model.push_to_hub(
            hf_repo,
            commit_message=f"Final checkpoint at step {trainer.state.global_step}",
            private=True,
        )
        tok.push_to_hub(hf_repo, private=True)
        print(f"Final model pushed to {hf_repo}")
    except Exception as e:
        print(f"Warning: Failed to push final model to HF: {e}")

    # Log final checkpoint as W&B artifact
    artifact = wandb.Artifact(
        name=f"model-{run_name}",
        type="model",
        description=f"Final GRPO model checkpoint ({ablation_name} ablation)",
    )
    artifact.add_dir(checkpoint_dir)
    wandb_run.log_artifact(artifact)

    print(f"Training complete! Model saved to {checkpoint_dir}")
    print(f"Model artifact logged to W&B: {wandb_run.url}")

    # Finish run
    wandb.finish()
