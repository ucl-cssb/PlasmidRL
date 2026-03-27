"""
Rejection sampling and best-of-N baselines for ICML revision.

Generates samples from base models (pre-training and SFT), scores them
with the full reward function, and reports pass rates and efficiency metrics.

Outputs are saved in analysis2-compatible format (CSV + FASTA per sequence).
"""

import datetime
import os
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import wandb
from vllm import LLM, SamplingParams

from src.ablations import get_ablation_config
from src.config import Config
from src.rewards.bioinformatics.scorer import Scorer


# Default models to evaluate
DEFAULT_MODELS = {
    "Base": "UCL-CSSB/PlasmidGPT",
    "SFT": "UCL-CSSB/PlasmidGPT-SFT",
}

# Default prompts (ATG + GFP cassette; extend with eval_prompts.csv when available)
DEFAULT_PROMPTS = None  # Will be loaded from config


def _get_prompts(cfg: Config) -> list[str]:
    """Get evaluation prompts."""
    return ["ATG", cfg.default_query]


def _clean_sequence(text: str) -> str:
    """Clean generated text to valid DNA."""
    return re.sub(r'[^ATCG]', '', text.upper().replace(" ", ""))


def _save_analysis2_format(
    df: pd.DataFrame,
    output_dir: str,
    model_label: str,
):
    """Save outputs in analysis2-compatible format.

    Creates:
        {output_dir}/outputs.csv  — columns: id, prompt, full
        {output_dir}/seq_{i}.fasta — one per sequence
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # CSV
    df.to_csv(out / "outputs.csv", index=False)

    # Individual FASTAs
    for i, row in df.iterrows():
        seq_id = row.get("id", f"seq_{i}")
        fasta = f">{seq_id}\n{row['full']}\n"
        (out / f"seq_{i}.fasta").write_text(fasta)

    print(f"[{model_label}] Saved {len(df)} sequences to {output_dir}")


def run_rejection_sampling(
    model_path: str,
    model_label: str,
    prompts: list[str],
    n_samples: int,
    scorer: Scorer,
    output_dir: str,
    gpu_util: float = 0.8,
) -> pd.DataFrame:
    """Generate samples and score them (rejection sampling).

    Args:
        model_path: HuggingFace model path.
        model_label: Label for logging (e.g., "Base", "SFT").
        prompts: List of prompt strings.
        n_samples: Total samples to generate (split evenly across prompts).
        scorer: Scorer instance for reward computation.
        output_dir: Directory to save outputs.
        gpu_util: vLLM GPU memory utilization.

    Returns:
        DataFrame with columns: id, prompt, completion, full, length, reward.
    """
    samples_per_prompt = n_samples // len(prompts)

    print(f"\n[{model_label}] Loading model: {model_path}")
    llm = LLM(model=model_path, gpu_memory_utilization=gpu_util, trust_remote_code=True)

    sampling_params = SamplingParams(
        max_tokens=256,
        temperature=1.1,
        top_p=0.90,
        stop_token_ids=[2],
    )

    # Expand prompts
    expanded = []
    for prompt in prompts:
        expanded.extend([prompt] * samples_per_prompt)

    print(f"[{model_label}] Generating {len(expanded)} samples ({samples_per_prompt} per prompt)...")
    t0 = time.time()
    outputs = llm.generate(expanded, sampling_params)
    gen_time = time.time() - t0

    # Process outputs
    records = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        completion = output.outputs[0].text.replace(" ", "")
        full = prompt + completion
        cleaned = _clean_sequence(full)

        # Score
        try:
            reward, components = scorer.score(cleaned)
        except Exception:
            reward = 0.0
            components = {}

        records.append({
            "id": f"seq_{i}",
            "prompt": prompt[:50],  # Truncate for readability
            "prompt_full": prompt,
            "completion": completion,
            "full": full,
            "length": len(cleaned),
            "reward": float(reward),
        })

    df = pd.DataFrame(records)

    # Save in analysis2 format
    _save_analysis2_format(df, output_dir, model_label)

    # Stats
    print(f"[{model_label}] Generation time: {gen_time:.1f}s ({len(df)/gen_time:.1f} samples/sec)")
    print(f"[{model_label}] Mean reward: {df['reward'].mean():.4f}")
    print(f"[{model_label}] Reward > 0.5: {(df['reward'] > 0.5).mean()*100:.1f}%")
    print(f"[{model_label}] Reward > 0.7: {(df['reward'] > 0.7).mean()*100:.1f}%")

    # Clean up GPU memory
    del llm

    return df


def run_best_of_n(
    model_path: str,
    model_label: str,
    prompts: list[str],
    group_size: int,
    target_per_prompt: int,
    scorer: Scorer,
    output_dir: str,
    gpu_util: float = 0.8,
) -> pd.DataFrame:
    """Best-of-N selection: generate groups, pick highest-reward per group.

    Args:
        model_path: HuggingFace model path.
        model_label: Label for logging.
        prompts: List of prompt strings.
        group_size: N for best-of-N (e.g., 16).
        target_per_prompt: Number of selected samples per prompt.
        scorer: Scorer instance.
        output_dir: Directory to save outputs.
        gpu_util: vLLM GPU memory utilization.

    Returns:
        DataFrame of selected (best) samples.
    """
    total_groups = target_per_prompt * len(prompts)
    total_samples = total_groups * group_size

    print(f"\n[{model_label}] Best-of-{group_size}: Loading model: {model_path}")
    llm = LLM(model=model_path, gpu_memory_utilization=gpu_util, trust_remote_code=True)

    sampling_params = SamplingParams(
        max_tokens=256,
        temperature=1.1,
        top_p=0.90,
        stop_token_ids=[2],
    )

    # Expand: for each prompt, generate target_per_prompt * group_size samples
    expanded = []
    prompt_indices = []
    for pi, prompt in enumerate(prompts):
        expanded.extend([prompt] * (target_per_prompt * group_size))
        prompt_indices.extend([pi] * (target_per_prompt * group_size))

    print(f"[{model_label}] Generating {len(expanded)} samples for best-of-{group_size} selection...")
    t0 = time.time()
    outputs = llm.generate(expanded, sampling_params)
    gen_time = time.time() - t0

    # Score all
    all_records = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        completion = output.outputs[0].text.replace(" ", "")
        full = prompt + completion
        cleaned = _clean_sequence(full)

        try:
            reward, _ = scorer.score(cleaned)
        except Exception:
            reward = 0.0

        all_records.append({
            "prompt_idx": prompt_indices[i],
            "prompt": prompt[:50],
            "prompt_full": prompt,
            "completion": completion,
            "full": full,
            "length": len(cleaned),
            "reward": float(reward),
        })

    df_all = pd.DataFrame(all_records)

    # Select best from each group
    selected = []
    for pi in range(len(prompts)):
        prompt_df = df_all[df_all["prompt_idx"] == pi].reset_index(drop=True)
        for g in range(target_per_prompt):
            start = g * group_size
            end = start + group_size
            group = prompt_df.iloc[start:end]
            if len(group) > 0:
                best = group.loc[group["reward"].idxmax()]
                selected.append(best)

    df_selected = pd.DataFrame(selected).reset_index(drop=True)
    df_selected["id"] = [f"seq_{i}" for i in range(len(df_selected))]

    # Save
    _save_analysis2_format(df_selected, output_dir, model_label)

    print(f"[{model_label}] Best-of-{group_size} time: {gen_time:.1f}s")
    print(f"[{model_label}] Selected {len(df_selected)} samples")
    print(f"[{model_label}] Mean reward (selected): {df_selected['reward'].mean():.4f}")
    print(f"[{model_label}] Reward > 0.5: {(df_selected['reward'] > 0.5).mean()*100:.1f}%")

    del llm
    return df_selected


def main(
    n_samples: int = 10000,
    best_of_n: int = 16,
    model_name: str | None = None,
    output_base: str = "results/baselines",
):
    """Run rejection sampling and best-of-N for all baseline models.

    Args:
        n_samples: Total rejection samples per model.
        best_of_n: Group size for best-of-N.
        model_name: If specified, only run this model. Otherwise run all defaults.
        output_base: Base directory for outputs.
    """
    cfg = Config()
    prompts = _get_prompts(cfg)
    reward_config = get_ablation_config("full_reward")
    scorer = Scorer(reward_config)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Determine which models to run
    if model_name:
        models = {"custom": model_name}
    else:
        models = DEFAULT_MODELS

    s3_base = "s3://phd-research-storage-1758274488/icml-revision/baselines"

    # Initialize W&B
    wandb_run = wandb.init(
        project="plasmid-rl-icml-revision",
        entity=cfg.wandb_entity,
        group="baselines",
        name=f"rejection-sampling-{timestamp}",
        tags=["icml-revision", "baselines", "rejection-sampling"],
        notes=(
            f"Rejection sampling & best-of-{best_of_n} baselines.\n\n"
            f"Generates {n_samples} samples per model from Base (UCL-CSSB/PlasmidGPT) "
            f"and SFT (UCL-CSSB/PlasmidGPT-SFT), scores with full reward function, "
            f"then selects best-of-{best_of_n} per group.\n\n"
            f"## Data Locations\n"
            f"- **S3 results**: {s3_base}/\n"
            f"  - `rejection_sampling/{{model}}/outputs.csv` — all scored samples\n"
            f"  - `rejection_sampling/{{model}}/seq_*.fasta` — individual sequences\n"
            f"  - `best_of_{best_of_n}/{{model}}/outputs.csv` — selected samples\n"
            f"  - `best_of_{best_of_n}/{{model}}/seq_*.fasta` — selected sequences\n"
            f"- **Prompts**: ATG + GFP cassette ({len(prompts)} prompts)\n"
        ),
        config={
            "n_samples": n_samples,
            "best_of_n": best_of_n,
            "models": models,
            "prompts": [p[:50] for p in prompts],
            "s3_results": s3_base,
        },
    )

    all_results = {}

    for label, model_path in models.items():
        # Rejection sampling
        rs_dir = os.path.join(output_base, f"rejection_sampling/{label}_{timestamp}")
        df_rs = run_rejection_sampling(
            model_path=model_path,
            model_label=f"{label}/RS",
            prompts=prompts,
            n_samples=n_samples,
            scorer=scorer,
            output_dir=rs_dir,
        )

        # Best-of-N
        bon_dir = os.path.join(output_base, f"best_of_{best_of_n}/{label}_{timestamp}")
        target_per_prompt = 500
        df_bon = run_best_of_n(
            model_path=model_path,
            model_label=f"{label}/BoN",
            prompts=prompts,
            group_size=best_of_n,
            target_per_prompt=target_per_prompt,
            scorer=scorer,
            output_dir=bon_dir,
        )

        # Log to W&B
        rs_metrics = {
            f"rs/{label}/mean_reward": df_rs["reward"].mean(),
            f"rs/{label}/pass_rate_0.5": (df_rs["reward"] > 0.5).mean(),
            f"rs/{label}/pass_rate_0.7": (df_rs["reward"] > 0.7).mean(),
            f"rs/{label}/n_samples": len(df_rs),
        }
        bon_metrics = {
            f"bon/{label}/mean_reward": df_bon["reward"].mean(),
            f"bon/{label}/pass_rate_0.5": (df_bon["reward"] > 0.5).mean(),
            f"bon/{label}/pass_rate_0.7": (df_bon["reward"] > 0.7).mean(),
            f"bon/{label}/n_selected": len(df_bon),
        }
        wandb.log({**rs_metrics, **bon_metrics})

        all_results[label] = {
            "rejection_sampling": df_rs,
            "best_of_n": df_bon,
        }

    # Summary table
    summary_rows = []
    for label, results in all_results.items():
        for method, df in results.items():
            summary_rows.append({
                "model": label,
                "method": method,
                "n_samples": len(df),
                "mean_reward": df["reward"].mean(),
                "pass_rate_0.5": (df["reward"] > 0.5).mean(),
                "pass_rate_0.7": (df["reward"] > 0.7).mean(),
                "mean_length": df["length"].mean(),
            })
    summary_df = pd.DataFrame(summary_rows)
    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))

    wandb.log({"summary": wandb.Table(dataframe=summary_df)})

    # Upload full dataframes as W&B artifacts (sequences + scores)
    for label, results in all_results.items():
        for method, df in results.items():
            artifact = wandb.Artifact(
                name=f"baseline-{label}-{method}".replace("_", "-"),
                type="baseline-sequences",
                description=f"{method} results for {label} model ({len(df)} sequences)",
            )
            # Save CSV to temp file and add to artifact
            csv_path = f"/tmp/{label}_{method}.csv"
            df.to_csv(csv_path, index=False)
            artifact.add_file(csv_path)
            wandb_run.log_artifact(artifact)

    wandb.finish()

    return all_results
