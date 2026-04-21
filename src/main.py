import click

from src.ablations import ABLATION_NAMES


@click.group()
def cli():
    """PlasmidRL: Reinforcement Learning for Plasmid Design"""
    pass


@cli.command("train-grpo")
def train_grpo():
    """Train model using Group Relative Policy Optimization (GRPO)"""
    from src.runners.grpo import run_grpo
    run_grpo()


@cli.command("train-ablation")
@click.option(
    "--config-name",
    required=True,
    type=click.Choice(ABLATION_NAMES),
    help="Ablation config name (see src/ablations.py)",
)
def train_ablation(config_name: str):
    """Train GRPO with a specific reward ablation configuration."""
    from src.runners.grpo import run_grpo
    run_grpo(ablation_name=config_name)


@cli.command("rejection-sampling")
@click.option("--n-samples", default=10000, help="Total samples to generate per model")
@click.option("--best-of-n", default=16, help="Group size for best-of-N selection")
@click.option("--model", "model_name", default=None, help="HF model path (default: both Base and SFT)")
def rejection_sampling(n_samples: int, best_of_n: int, model_name: str | None):
    """Run rejection sampling and best-of-N baselines."""
    from src.runners.rejection_sampling import main as rs_main
    rs_main(n_samples=n_samples, best_of_n=best_of_n, model_name=model_name)


@cli.command("generate-samples")
def generate_samples():
    """Generate samples using vLLM"""
    from src.runners.generate_samples import main
    df = main()
    print(df.head())


@cli.command("convert-checkpoint")
@click.option("--checkpoint-path", required=True, help="S3 path to checkpoint (e.g., s3://bucket/path/to/checkpoint)")
@click.option("--hf-repo", required=True, help="HuggingFace repository path (e.g., username/repo-name)")
def convert_checkpoint(checkpoint_path: str, hf_repo: str):
    """Convert VERL/GRPO checkpoint to HuggingFace format and upload"""
    from src.utils.model_utils import checkpoint_to_huggingface, s3_client

    click.echo(f"Converting checkpoint from {checkpoint_path}")
    click.echo(f"Target HuggingFace repo: {hf_repo}")
    url = checkpoint_to_huggingface(s3_client, checkpoint_path, hf_repo)
    click.echo(f"Model available at: {url}")


if __name__ == "__main__":
    cli()
