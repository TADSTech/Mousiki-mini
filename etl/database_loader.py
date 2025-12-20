"""
Production-grade Database Loader for Mousiki.

Handles loading processed data into PostgreSQL with connection pooling,
transaction management, and error recovery.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd
import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import execute_values

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging_config import error, neutral, success, warning
from utils.config import settings


class DatabaseError(Exception):
    """Base exception for database operations."""
    pass


class DatabaseLoader:
    """
    Production-ready database loader with connection pooling.
    
    Features:
    - Connection pooling for efficient resource usage
    - Transaction management with rollback
    - Batch inserts for performance
    - Duplicate handling (upsert)
    - Comprehensive error logging
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        min_conn: int = 1,
        max_conn: int = 10
    ):
        """
        Initialize database loader with connection pool.
        
        Args:
            host: Database host.
            port: Database port.
            database: Database name.
            user: Database user.
            password: Database password.
            min_conn: Minimum connections in pool.
            max_conn: Maximum connections in pool.
        """
        self.host = host or settings.DB_HOST
        self.port = port or settings.DB_PORT
        self.database = database or settings.DB_NAME
        self.user = user or settings.DB_USER
        self.password = password or settings.DB_PASSWORD
        
        self.connection_pool = None
        self._init_connection_pool(min_conn, max_conn)
    
    def _init_connection_pool(self, min_conn: int, max_conn: int) -> None:
        """Initialize PostgreSQL connection pool."""
        try:
            neutral(f"Initializing connection pool (min={min_conn}, max={max_conn})", "DB")
            
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                min_conn,
                max_conn,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            
            success(f"Connected to database: {self.database}@{self.host}:{self.port}", "DB")
            
        except Exception as e:
            error(f"Failed to initialize connection pool: {str(e)}", "DB")
            raise DatabaseError("Connection pool initialization failed") from e
    
    def get_connection(self):
        """Get connection from pool."""
        try:
            return self.connection_pool.getconn()
        except Exception as e:
            error(f"Failed to get connection from pool: {str(e)}", "DB")
            raise DatabaseError("Could not acquire connection") from e
    
    def return_connection(self, conn) -> None:
        """Return connection to pool."""
        if conn:
            self.connection_pool.putconn(conn)
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> None:
        """
        Execute a query with transaction management.
        
        Args:
            query: SQL query string.
            params: Query parameters.
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            cursor.close()
            
        except Exception as e:
            if conn:
                conn.rollback()
            error(f"Query execution failed: {str(e)}", "DB")
            raise DatabaseError("Query execution failed") from e
            
        finally:
            self.return_connection(conn)
    
    def load_tracks(
        self, 
        df: pd.DataFrame, 
        batch_size: int = 1000,
        on_conflict: str = "update"
    ) -> int:
        """
        Load tracks into database with batch processing.
        
        Args:
            df: DataFrame with track data.
            batch_size: Number of records per batch.
            on_conflict: Action on conflict ('update' or 'skip').
            
        Returns:
            Number of tracks loaded.
        """
        required_cols = ["title", "artist"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        neutral(f"Loading {len(df):,} tracks to database...", "DB")
        
        # Convert to records and prepare for bulk insert
        records = []
        for _, row in df.iterrows():
            records.append((
                row.get("title"),
                row.get("artist"),
                row.get("album"),
                row.get("genre"),
                row.get("duration"),
                row.get("year"),
                row.get("popularity", 0.0)
            ))
        
        # Deduplicate records by (title, artist) - keep first occurrence
        seen = set()
        deduplicated_records = []
        duplicate_count = 0
        
        for record in records:
            key = (record[0], record[1])  # (title, artist)
            if key not in seen:
                seen.add(key)
                deduplicated_records.append(record)
            else:
                duplicate_count += 1
        
        if duplicate_count > 0:
            warning(f"Found {duplicate_count:,} duplicate records in batch, keeping first occurrence", "DB")
        
        records = deduplicated_records
        
        # Batch insert
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if on_conflict == "update":
                query = """
                    INSERT INTO tracks (title, artist, album, genre, duration, year, popularity)
                    VALUES %s
                    ON CONFLICT (title, artist) DO UPDATE SET
                        album = EXCLUDED.album,
                        genre = EXCLUDED.genre,
                        duration = EXCLUDED.duration,
                        year = EXCLUDED.year,
                        popularity = EXCLUDED.popularity,
                        updated_at = CURRENT_TIMESTAMP
                """
            else:
                query = """
                    INSERT INTO tracks (title, artist, album, genre, duration, year, popularity)
                    VALUES %s
                    ON CONFLICT (title, artist) DO NOTHING
                """
            
            # Add unique constraint if not exists
            try:
                cursor.execute("""
                    ALTER TABLE tracks ADD CONSTRAINT tracks_title_artist_unique 
                    UNIQUE (title, artist)
                """)
                conn.commit()
            except psycopg2.errors.DuplicateTable:
                conn.rollback()
            
            # Execute batch insert
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                execute_values(cursor, query, batch)
                neutral(f"Loaded batch {i//batch_size + 1}/{(len(records)-1)//batch_size + 1}", "DB")
            
            conn.commit()
            cursor.close()
            
            success(f"Successfully loaded {len(records):,} tracks", "DB")
            return len(records)
            
        except Exception as e:
            if conn:
                conn.rollback()
            error(f"Failed to load tracks: {str(e)}", "DB")
            raise DatabaseError("Track loading failed") from e
            
        finally:
            self.return_connection(conn)
    
    def load_interactions(
        self,
        df: pd.DataFrame,
        batch_size: int = 1000
    ) -> int:
        """
        Load user interactions into database.
        
        Args:
            df: DataFrame with interaction data.
            batch_size: Number of records per batch.
            
        Returns:
            Number of interactions loaded.
        """
        required_cols = ["user_id", "track_id", "interaction_type"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        neutral(f"Loading {len(df):,} interactions to database...", "DB")
        
        # Prepare data
        records = []
        for _, row in df.iterrows():
            records.append((
                int(row["user_id"]),
                int(row["track_id"]),
                row["interaction_type"],
                row.get("duration"),
                row.get("timestamp", datetime.now())
            ))
        
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            query = """
                INSERT INTO interactions (user_id, track_id, interaction_type, duration, timestamp)
                VALUES %s
            """
            
            # Execute batch insert
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                execute_values(cursor, query, batch)
                neutral(f"Loaded batch {i//batch_size + 1}/{(len(records)-1)//batch_size + 1}", "DB")
            
            conn.commit()
            cursor.close()
            
            success(f"Successfully loaded {len(records):,} interactions", "DB")
            return len(records)
            
        except Exception as e:
            if conn:
                conn.rollback()
            error(f"Failed to load interactions: {str(e)}", "DB")
            raise DatabaseError("Interaction loading failed") from e
            
        finally:
            self.return_connection(conn)
    
    def initialize_schema(self, schema_file: str = "./api/db/schema.sql") -> None:
        """
        Initialize database schema from SQL file.
        
        Args:
            schema_file: Path to SQL schema file.
        """
        schema_path = Path(schema_file)
        
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_file}")
        
        neutral(f"Initializing database schema from {schema_file}", "DB")
        
        conn = None
        try:
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(schema_sql)
            conn.commit()
            cursor.close()
            
            success("Database schema initialized successfully", "DB")
            
        except Exception as e:
            if conn:
                conn.rollback()
            error(f"Schema initialization failed: {str(e)}", "DB")
            raise DatabaseError("Schema initialization failed") from e
            
        finally:
            self.return_connection(conn)
    
    def close(self) -> None:
        """Close all connections in the pool."""
        if self.connection_pool:
            self.connection_pool.closeall()
            neutral("Database connection pool closed", "DB")
