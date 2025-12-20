"""Generate sample interactions for CF model training."""

import random
import psycopg2
from datetime import datetime, timedelta
from collections import defaultdict

# Database connection
conn = psycopg2.connect(
    host="localhost",
    database="mousiki",
    user="mousiki_user",
    password="mousiki_password"
)

cursor = conn.cursor()

# First, create sample users if they don't exist
print("Creating sample users...")
num_users = 100
for i in range(num_users):
    username = f"user_{i+1}"
    email = f"user{i+1}@example.com"
    
    cursor.execute("""
        INSERT INTO users (username, email)
        VALUES (%s, %s)
        ON CONFLICT (username) DO NOTHING
    """, (username, email))

conn.commit()

# Get existing users and tracks
cursor.execute("SELECT user_id FROM users LIMIT 200")
users = [row[0] for row in cursor.fetchall()]

cursor.execute("SELECT track_id FROM tracks LIMIT 1000")
tracks = [row[0] for row in cursor.fetchall()]

if not users or not tracks:
    print("❌ No users or tracks found in database. Run ETL pipeline first.")
    conn.close()
    exit(1)

print(f"Found {len(users)} users and {len(tracks)} tracks")

# Generate synthetic interactions
interactions = []
interaction_types = ['play', 'like', 'skip', 'share', 'playlist_add']

# For each user, create multiple interactions
for user_id in users:
    # Random number of tracks each user interacts with
    num_tracks = random.randint(5, 50)
    user_tracks = random.sample(tracks, min(num_tracks, len(tracks)))
    
    for track_id in user_tracks:
        # Generate 1-5 interactions per user-track pair
        num_interactions = random.randint(1, 5)
        base_timestamp = datetime.now() - timedelta(days=random.randint(1, 365))
        
        for _ in range(num_interactions):
            interaction_type = random.choice(interaction_types)
            # Skips are more common, likes are less common
            if interaction_type == 'skip' and random.random() > 0.3:
                continue
            if interaction_type == 'like' and random.random() > 0.2:
                continue
            
            timestamp = base_timestamp + timedelta(hours=random.randint(1, 24))
            
            interactions.append({
                'user_id': user_id,
                'track_id': track_id,
                'interaction_type': interaction_type,
                'duration': random.randint(30, 300) if interaction_type == 'play' else 0,
                'timestamp': timestamp
            })

print(f"Generated {len(interactions)} interactions")

# Insert in batches
batch_size = 1000
for i in range(0, len(interactions), batch_size):
    batch = interactions[i:i+batch_size]
    
    insert_query = """
    INSERT INTO interactions (user_id, track_id, interaction_type, duration, timestamp)
    VALUES (%s, %s, %s, %s, %s)
    """
    
    cursor.executemany(insert_query, [
        (
            inter['user_id'],
            inter['track_id'],
            inter['interaction_type'],
            inter['duration'],
            inter['timestamp']
        )
        for inter in batch
    ])
    
    print(f"  Inserted {min(i+batch_size, len(interactions))}/{len(interactions)}")

conn.commit()
conn.close()

print(f"\n✓ Successfully inserted {len(interactions)} interactions!")
print(f"  Users: {len(users)}")
print(f"  Tracks: {len(tracks)}")
