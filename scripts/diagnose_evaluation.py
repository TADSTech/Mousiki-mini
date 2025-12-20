"""
Diagnostic script to understand evaluation results.

Analyzes why model performance is low and provides insights.
"""

import json
import pandas as pd
from pathlib import Path
import psycopg2
from datetime import datetime


def connect_db():
    """Connect to database."""
    return psycopg2.connect(
        dbname="mousiki",
        user="mousiki_user",
        password="mousiki_password",
        host="localhost",
        port=5432
    )


def analyze_data_coverage():
    """Analyze interaction data coverage."""
    conn = connect_db()
    cur = conn.cursor()
    
    print("="*60)
    print("Data Coverage Analysis")
    print("="*60)
    
    # Total users and tracks
    cur.execute("SELECT COUNT(DISTINCT user_id) FROM interactions")
    n_users = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(DISTINCT track_id) FROM interactions")
    n_tracks_interacted = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM tracks")
    n_tracks_total = cur.fetchone()[0]
    
    print(f"\nUsers: {n_users}")
    print(f"Tracks with interactions: {n_tracks_interacted}")
    print(f"Total tracks in DB: {n_tracks_total}")
    print(f"Coverage: {n_tracks_interacted/n_tracks_total*100:.2f}%")
    
    # Sparsity
    cur.execute("SELECT COUNT(*) FROM interactions")
    n_interactions = cur.fetchone()[0]
    
    sparsity = 1 - (n_interactions / (n_users * n_tracks_total))
    print(f"\nTotal interactions: {n_interactions}")
    print(f"Sparsity: {sparsity*100:.4f}%")
    
    # Interactions per user
    cur.execute("""
        SELECT 
            COUNT(*) as n_interactions,
            COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage
        FROM interactions
        GROUP BY user_id
        ORDER BY n_interactions DESC
        LIMIT 10
    """)
    
    print("\nTop 10 users by interactions:")
    for row in cur.fetchall():
        print(f"  {row[0]} interactions ({row[1]:.1f}%)")
    
    # Average interactions per user
    cur.execute("""
        SELECT AVG(cnt), MIN(cnt), MAX(cnt)
        FROM (SELECT COUNT(*) as cnt FROM interactions GROUP BY user_id) t
    """)
    avg, min_int, max_int = cur.fetchone()
    print(f"\nInteractions per user: avg={avg:.1f}, min={min_int}, max={max_int}")
    
    # Interaction types
    cur.execute("""
        SELECT interaction_type, COUNT(*), 
               COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage
        FROM interactions
        GROUP BY interaction_type
        ORDER BY COUNT(*) DESC
    """)
    
    print("\nInteraction types:")
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]} ({row[2]:.1f}%)")
    
    cur.close()
    conn.close()


def analyze_evaluation_results():
    """Analyze evaluation results in detail."""
    results_dir = Path("./evaluation_results")
    
    if not results_dir.exists():
        print("No evaluation results found")
        return
    
    # Load latest results
    comparison_files = list(results_dir.glob("model_comparison_*.csv"))
    if not comparison_files:
        print("No comparison files found")
        return
    
    latest = sorted(comparison_files)[-1]
    timestamp = latest.stem.split("_")[-1]
    
    print("\n" + "="*60)
    print("Evaluation Results Analysis")
    print("="*60)
    
    # Load comparison
    comparison_df = pd.read_csv(latest)
    print("\nModel Performance:")
    print(comparison_df.to_string(index=False))
    
    # Load per-user results
    for model_name in comparison_df['model']:
        results_file = results_dir / f"{model_name}_results_{timestamp}.csv"
        if results_file.exists():
            df = pd.read_csv(results_file)
            
            print(f"\n{model_name} Details:")
            print(f"  Users evaluated: {len(df)}")
            print(f"  Users with hits@20: {(df['hit_rate@20'] > 0).sum()}")
            
            if 'method' in df.columns:
                method_counts = df['method'].value_counts()
                print(f"  Methods used:")
                for method, count in method_counts.items():
                    print(f"    {method}: {count} ({count/len(df)*100:.1f}%)")
            
            # Show users with non-zero performance
            good_users = df[df['hit_rate@20'] > 0]
            if len(good_users) > 0:
                print(f"\n  Users with hits (sample):")
                for _, row in good_users.head(5).iterrows():
                    print(f"    User {row['user_id']}: hit_rate@20={row['hit_rate@20']:.3f}")


def analyze_temporal_split():
    """Analyze how temporal split affected data."""
    conn = connect_db()
    cur = conn.cursor()
    
    print("\n" + "="*60)
    print("Temporal Split Analysis")
    print("="*60)
    
    # For each user, show train/test split
    cur.execute("""
        WITH user_stats AS (
            SELECT 
                user_id,
                COUNT(*) as total,
                MIN(timestamp) as first_ts,
                MAX(timestamp) as last_ts
            FROM interactions
            GROUP BY user_id
        )
        SELECT 
            AVG(total) as avg_interactions,
            MIN(total) as min_interactions,
            MAX(total) as max_interactions,
            COUNT(*) as n_users_with_5plus
        FROM user_stats
        WHERE total >= 5
    """)
    
    result = cur.fetchone()
    print(f"\nUsers with ≥5 interactions: {result[3]}")
    print(f"  Avg interactions: {result[0]:.1f}")
    print(f"  Min/Max: {result[1]}/{result[2]}")
    
    # Time range
    cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM interactions")
    min_ts, max_ts = cur.fetchone()
    print(f"\nTime range: {min_ts} to {max_ts}")
    
    # Calculate 80/20 split point for a sample user
    cur.execute("""
        SELECT user_id, COUNT(*) as cnt
        FROM interactions
        GROUP BY user_id
        ORDER BY cnt DESC
        LIMIT 1
    """)
    top_user, top_count = cur.fetchone()
    
    cur.execute("""
        SELECT timestamp
        FROM interactions
        WHERE user_id = %s
        ORDER BY timestamp
        LIMIT 1 OFFSET %s
    """, (top_user, int(top_count * 0.8)))
    
    split_point = cur.fetchone()
    if split_point:
        print(f"\nSample split (user {top_user} with {top_count} interactions):")
        print(f"  Train: 80% ({int(top_count * 0.8)} interactions)")
        print(f"  Test: 20% ({int(top_count * 0.2)} interactions)")
        print(f"  Split at: {split_point[0]}")
    
    cur.close()
    conn.close()


def provide_recommendations():
    """Provide recommendations for improvement."""
    print("\n" + "="*60)
    print("Recommendations for Improvement")
    print("="*60)
    
    recommendations = [
        ("1. Generate More Interactions", 
         "Current: ~5,235 interactions for 100 users\n"
         "   Recommended: 50,000+ interactions (500+ per user)\n"
         "   Command: python scripts/generate_sample_interactions.py --num_users 500 --interactions_per_user 200"),
        
        ("2. Improve Interaction Quality",
         "Use more 'like' and 'playlist_add' interactions (positive signals)\n"
         "   Current distribution may be too random\n"
         "   Add preference modeling to interaction generator"),
        
        ("3. Reduce Track Space",
         "57,648 total tracks is very large for 100 users\n"
         "   Consider using a subset of popular tracks for synthetic data\n"
         "   This will increase coverage and reduce sparsity"),
        
        ("4. Alternative Evaluation",
         "Low recall is expected with huge track catalogs\n"
         "   Use ranking metrics: MRR, NDCG (already computed)\n"
         "   Consider domain-specific metrics (artist diversity, genre coverage)\n"
         "   Evaluate on real user feedback (online A/B testing)"),
        
        ("5. Model Improvements",
         "CBF: Tune similarity threshold and weighting\n"
         "   CF: Train with more epochs, tune hyperparameters\n"
         "   Hybrid: Experiment with different weight combinations"),
        
        ("6. Use Real Data",
         "Synthetic data doesn't capture real user preferences\n"
         "   Consider using public datasets (LastFM, Million Song)\n"
         "   Or collect real user interactions")
    ]
    
    for title, desc in recommendations:
        print(f"\n{title}")
        print("  " + desc.replace("\n", "\n  "))


def main():
    """Run diagnostics."""
    print("Mousiki Evaluation Diagnostics")
    print()
    
    analyze_data_coverage()
    analyze_temporal_split()
    analyze_evaluation_results()
    provide_recommendations()
    
    print("\n" + "="*60)
    print("Diagnostics Complete")
    print("="*60)


if __name__ == "__main__":
    main()
