#!/bin/bash
# Setup PostgreSQL tunnel for Google Colab access

echo "==================================================================="
echo "PostgreSQL Tunnel Setup for Google Colab"
echo "==================================================================="

# Configuration
PG_PORT=5432
TUNNEL_SERVICE=${1:-"ngrok"}  # ngrok, serveo, localhost.run

case $TUNNEL_SERVICE in
    ngrok)
        echo "Using ngrok..."
        echo ""
        echo "1. Install ngrok if not already installed:"
        echo "   brew install ngrok    # macOS"
        echo "   snap install ngrok    # Linux"
        echo ""
        echo "2. Sign up at https://ngrok.com and get your authtoken"
        echo ""
        echo "3. Configure authtoken:"
        echo "   ngrok config add-authtoken YOUR_TOKEN"
        echo ""
        echo "Starting ngrok tunnel..."
        ngrok tcp $PG_PORT
        ;;
        
    serveo)
        echo "Using serveo.net (no signup required)..."
        echo ""
        echo "Starting SSH tunnel via serveo.net..."
        ssh -R 5432:localhost:5432 serveo.net
        echo ""
        echo "Copy the URL provided above and use in Colab:"
        echo "  DB_HOST = 'serveo.net'"
        echo "  DB_PORT = <port shown above>"
        ;;
        
    localhost.run)
        echo "Using localhost.run (no signup required)..."
        echo ""
        echo "Starting SSH tunnel via localhost.run..."
        ssh -R 80:localhost:5432 ssh.localhost.run
        ;;
        
    *)
        echo "Unknown tunnel service: $TUNNEL_SERVICE"
        echo "Supported services: ngrok, serveo, localhost.run"
        exit 1
        ;;
esac
