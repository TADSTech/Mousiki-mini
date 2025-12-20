#!/bin/bash

# Supabase ETL Setup Script
# This script sets up the environment and runs the ETL pipeline

echo "=========================================="
echo "Mousiki Supabase ETL Setup"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install -q supabase pandas python-dotenv

# Check if data file exists
if [ ! -f "data/processed/tracks_preprocessed.csv" ]; then
    echo "ERROR: data/processed/tracks_preprocessed.csv not found!"
    exit 1
fi

echo ""
echo "Data file found: data/processed/tracks_preprocessed.csv"
wc -l data/processed/tracks_preprocessed.csv

# Load environment variables from frontend/.env
if [ -f "frontend/.env" ]; then
    echo ""
    echo "Loading Supabase credentials from frontend/.env..."
    export $(grep -v '^#' frontend/.env | xargs)
else
    echo "WARNING: frontend/.env not found!"
fi

# Check credentials
if [ -z "$VITE_SUPABASE_URL" ] || [ -z "$VITE_SUPABASE_ANON_KEY" ]; then
    echo "ERROR: Supabase credentials not found in environment!"
    echo "Please ensure frontend/.env contains:"
    echo "  VITE_SUPABASE_URL=your_url"
    echo "  VITE_SUPABASE_ANON_KEY=your_key"
    exit 1
fi

echo ""
echo "Supabase URL: $VITE_SUPABASE_URL"
echo "Supabase Key: ${VITE_SUPABASE_ANON_KEY:0:20}..."

echo ""
echo "=========================================="
echo "Running ETL Pipeline..."
echo "=========================================="
echo ""

# Set ETL limit (change to larger number or remove for full load)
export ETL_LIMIT=${ETL_LIMIT:-10000}
echo "ETL_LIMIT: $ETL_LIMIT tracks"
echo ""

# Run the ETL
python3 -m etl.supabase_loader

echo ""
echo "=========================================="
echo "ETL Complete!"
echo "=========================================="
