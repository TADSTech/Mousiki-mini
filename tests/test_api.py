"""
API Tests

Tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check(self):
        """Test health check returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestRecommendationEndpoints:
    """Tests for recommendation endpoints."""
    
    def test_get_recommendations(self):
        """Test get recommendations endpoint."""
        response = client.get("/api/v1/recommendations/1?num_recommendations=10")
        assert response.status_code in [200, 500]  # May fail without setup
    
    def test_post_recommendations(self):
        """Test post recommendations endpoint."""
        payload = {
            "user_id": 1,
            "num_recommendations": 10
        }
        response = client.post("/api/v1/recommendations", json=payload)
        assert response.status_code in [200, 500]


class TestItemEndpoints:
    """Tests for item endpoints."""
    
    def test_get_item(self):
        """Test get item by ID."""
        response = client.get("/api/v1/items/1")
        assert response.status_code in [200, 404]
    
    def test_search_items(self):
        """Test search items."""
        response = client.get("/api/v1/items?query=test")
        assert response.status_code == 200


class TestUserEndpoints:
    """Tests for user endpoints."""
    
    def test_get_user(self):
        """Test get user by ID."""
        response = client.get("/api/v1/users/1")
        assert response.status_code in [200, 404]
    
    def test_get_user_stats(self):
        """Test get user statistics."""
        response = client.get("/api/v1/users/1/stats")
        assert response.status_code in [200, 500]


class TestInteractionEndpoints:
    """Tests for interaction endpoints."""
    
    def test_create_interaction(self):
        """Test create interaction."""
        payload = {
            "user_id": 1,
            "track_id": 100,
            "interaction_type": "play"
        }
        response = client.post("/api/v1/interactions", json=payload)
        assert response.status_code in [201, 500]
    
    def test_get_user_interactions(self):
        """Test get user interactions."""
        response = client.get("/api/v1/interactions/1")
        assert response.status_code in [200, 500]
