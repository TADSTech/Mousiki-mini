#!/usr/bin/env python3
"""
Mousiki Mini CLI
The main entry point for the simplified Mousiki Music Recommendation System.
"""

import sys
import os
from pathlib import Path
import logging
import typer
from typing import Optional
from rich.console import Console
from rich.table import Table

# Add current directory to path so we can import models
sys.path.append(os.getcwd())

from models.hybrid.hybrid_recommender import HybridRecommender

app = typer.Typer(help="Mousiki Mini - Music Recommendation System CLI")
console = Console()
logging.basicConfig(level=logging.ERROR)  # Suppress info logs for CLI output

def get_latest_file(directory: Path, pattern: str) -> Optional[Path]:
    """Find the latest file matching a pattern in a directory."""
    try:
        files = list(directory.glob(pattern))
        if not files:
            return None
        return max(files, key=os.path.getctime)
    except Exception:
        return None

def find_model_paths():
    """Auto-discover model paths."""
    base_dir = Path("models")
    
    # CBF paths
    content_dir = base_dir / "content" / "model"
    cbf_path = get_latest_file(content_dir, "*.npz")
    
    # CF paths
    cf_dir = base_dir / "collaborative" / "model_weights"
    cf_model = get_latest_file(cf_dir, "*.pt")
    cf_mappings = get_latest_file(cf_dir, "*.pkl")
    
    return cbf_path, cf_model, cf_mappings

@app.command()
def setup():
    """
    Verify setup and model availability.
    """
    console.print("[bold blue]Checking Mousiki Setup...[/bold blue]")
    
    cbf_path, cf_model, cf_mappings = find_model_paths()
    
    if cbf_path:
        console.print(f"[green]✓ Found CBF Similarity Matrix:[/green] {cbf_path.name}")
    else:
        console.print("[red]✗ Missing CBF Similarity Matrix[/red]")
        
    if cf_model:
        console.print(f"[green]✓ Found CF Model:[/green] {cf_model.name}")
    else:
        console.print("[red]✗ Missing CF Model[/red]")
        
    if cf_mappings:
        console.print(f"[green]✓ Found CF Mappings:[/green] {cf_mappings.name}")
    else:
        console.print("[red]✗ Missing CF Mappings[/red]")
        
    if all([cbf_path, cf_model, cf_mappings]):
        console.print("\n[bold green]Ready to recommend![/bold green]")
    else:
        console.print("\n[bold yellow]Some models are missing. You may need to run training scripts.[/bold yellow]")

@app.command()
def list_users(limit: int = 50):
    """List valid User IDs from the model."""
    _, _, cf_mappings = find_model_paths()
    if not cf_mappings:
        console.print("[red]CF Mappings not found.[/red]")
        return

    import pickle
    with open(cf_mappings, 'rb') as f:
        mappings = pickle.load(f)
        user_ids = list(mappings['user_id_map'].keys())
        
    console.print(f"[bold]Found {len(user_ids)} users. Showing first {limit}:[/bold]")
    console.print(", ".join(map(str, user_ids[:limit])))


@app.command()
def recommend(
    user_id: int = typer.Option(..., help="User ID to generate recommendations for"),
    n: int = typer.Option(10, help="Number of recommendations"),
    cbf: bool = typer.Option(True, help="Use Content-Based Filtering"),
    cf: bool = typer.Option(True, help="Use Collaborative Filtering"),
    history: str = typer.Option(None, help="Comma-separated list of track IDs for history override")
):
    """
    Generate recommendations for a user.
    """
    cbf_path, cf_model, cf_mappings = find_model_paths()
    
    if not all([cbf_path, cf_model, cf_mappings]):
        console.print("[red]Error: Models not found. Run 'setup' to check status.[/red]")
        raise typer.Exit(code=1)
        
    try:
        recommender = HybridRecommender(
            cbf_similarity_path=str(cbf_path),
            cf_model_path=str(cf_model),
            cf_mappings_path=str(cf_mappings),
            device="cpu"  # Force CPU for stability in demo
        )
        
        # Parse history override
        history_list = None
        if history:
            try:
                history_list = [int(x.strip()) for x in history.split(",")]
                console.print(f"[dim]Using history override: {history_list}[/dim]")
            except ValueError:
                console.print("[yellow]Invalid history format. Ignoring.[/yellow]")

        with console.status(f"[bold green]Generating {n} recommendations for User {user_id}..."):
            result = recommender.recommend(
                user_id=user_id,
                n_recommendations=n,
                use_cbf=cbf,
                use_cf=cf,
                user_history_override=history_list
            )
            
        console.print(f"\n[bold]Recommendations for User {user_id}[/bold] (Method: [cyan]{result.method}[/cyan])")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Track ID", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Source", justify="right")
        
        for i, (track_id, score) in enumerate(result.recommendations, 1):
            # Determine source details
            sources = []
            if track_id in result.cbf_scores: sources.append("CBF")
            if track_id in result.cf_scores: sources.append("CF")
            if track_id in result.hybrid_scores: sources.append("Hybrid")
            if not sources: sources.append("Pop")
            
            table.add_row(
                str(i), 
                str(track_id), 
                f"{score:.4f}", 
                ", ".join(sources)
            )
            
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
