#!/usr/bin/env bash
#
# Mousiki Database Setup Script
#
# This script sets up the PostgreSQL database for Mousiki

set -e

echo "🎵 Mousiki Database Setup"
echo "========================="
echo

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo "❌ PostgreSQL is not installed"
    echo "Install with:"
    echo "  Ubuntu/Debian: sudo apt-get install postgresql postgresql-contrib"
    echo "  macOS: brew install postgresql"
    exit 1
fi

# Check if PostgreSQL is running
if ! pgrep -x postgres > /dev/null; then
    echo "⚠️  PostgreSQL is not running"
    echo "Start with:"
    echo "  Linux: sudo systemctl start postgresql"
    echo "  macOS: brew services start postgresql"
    exit 1
fi

echo "✅ PostgreSQL is installed and running"
echo

# Database configuration
DB_NAME=${DB_NAME:-mousiki}
DB_USER=${DB_USER:-mousiki_user}
DB_PASSWORD=${DB_PASSWORD:-mousiki_password}

echo "Database Configuration:"
echo "  Name: $DB_NAME"
echo "  User: $DB_USER"
echo

# Create database and user
echo "Creating database and user..."

sudo -u postgres psql << EOF
-- Create database
CREATE DATABASE $DB_NAME;

-- Create user
CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;

-- Grant schema privileges (PostgreSQL 15+)
\c $DB_NAME
GRANT ALL ON SCHEMA public TO $DB_USER;

\q
EOF

if [ $? -eq 0 ]; then
    echo "✅ Database and user created successfully"
else
    echo "⚠️  Database/user may already exist (this is OK)"
fi

echo

# Initialize schema
if [ -f "api/db/schema.sql" ]; then
    echo "Initializing database schema..."
    PGPASSWORD=$DB_PASSWORD psql -U $DB_USER -d $DB_NAME -f api/db/schema.sql
    
    if [ $? -eq 0 ]; then
        echo "✅ Schema initialized successfully"
    else
        echo "❌ Failed to initialize schema"
        exit 1
    fi
else
    echo "⚠️  Schema file not found: api/db/schema.sql"
fi

echo

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    
    # Update database credentials in .env
    sed -i "s/DB_NAME=.*/DB_NAME=$DB_NAME/" .env
    sed -i "s/DB_USER=.*/DB_USER=$DB_USER/" .env
    sed -i "s/DB_PASSWORD=.*/DB_PASSWORD=$DB_PASSWORD/" .env
    
    echo "✅ .env file created"
else
    echo "ℹ️  .env file already exists"
fi

echo
echo "🎉 Database setup complete!"
echo
echo "You can now run the ETL pipeline:"
echo "  python -m etl.pipeline"
echo
echo "Or test the connection:"
echo "  psql -U $DB_USER -d $DB_NAME"
echo
