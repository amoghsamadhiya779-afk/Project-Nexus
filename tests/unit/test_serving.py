#!/usr/bin/env python3
"""
=============================================================================
Nexus Serving Gateway - Unit Tests
Validates the FastAPI gateway endpoints using Starlette TestClient and 
properly mocks internal ML singletons to pass Pydantic schema validation.
=============================================================================
"""

import pytest
import numpy as np
from unittest.mock import patch
from fastapi.testclient import TestClient

# Import the app directly (the singletons will fall back to simulation mode safely)
from services.serving.gateway.app import app

client = TestClient(app)

@patch('services.serving.gateway.app.cache.get')
@patch('services.serving.gateway.app.cache.set')
@patch('services.serving.gateway.app.fetcher.fetch_user_features')
@patch('services.serving.gateway.app.model_server.predict_user_embedding')
@patch('services.serving.gateway.app.search_engine.retrieve_candidates')
@patch('services.serving.gateway.app.fetcher.fetch_item_features')
@patch('services.serving.gateway.app.model_server.score_ranking_batch')
def test_recommendation_endpoint_success(
    mock_score, mock_fetch_items, mock_retrieve, mock_predict_user, 
    mock_fetch_user, mock_cache_set, mock_cache_get
):
    """Validates the Recommendation API returns correctly formatted schema data."""
    # 1. Setup mocks to return valid data types matching the Pydantic schema
    mock_cache_get.return_value = None
    mock_fetch_user.return_value = [{"user_id": "user_100", "user_view_count": 5}]
    mock_predict_user.return_value = np.array([0.1, 0.2, 0.3])
    mock_retrieve.return_value = (["item_1", "item_2"], [0.9, 0.8])
    mock_fetch_items.return_value = [{"item_id": "item_1"}, {"item_id": "item_2"}]
    # Return two numpy arrays representing CTR and CVR scores
    mock_score.return_value = (np.array([0.85, 0.92]), np.array([0.15, 0.22]))

    # 2. Execute the request
    valid_payload = {"user_id": "user_100", "k_candidates": 2, "use_cache": False}
    response = client.post("/recommend", json=valid_payload)
    
    # 3. Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "user_100"
    assert len(data["items"]) == 2
    assert "ctr_predictions" in data
    assert "cvr_predictions" in data
    assert data["cached"] is False


@patch('services.serving.gateway.app.fetcher.fetch_user_features')
@patch('services.serving.gateway.app.model_server.predict_user_embedding')
@patch('services.serving.gateway.app.search_engine.retrieve_candidates')
@patch('services.serving.gateway.app.fetcher.fetch_item_features')
@patch('services.serving.gateway.app.search_engine.rescore_with_ltr')
def test_search_endpoint_success(
    mock_rescore, mock_fetch_items, mock_retrieve, mock_predict_user, mock_fetch_user
):
    """Validates the Semantic Search API schema structure."""
    mock_fetch_user.return_value = [{"user_id": "user_1"}]
    mock_predict_user.return_value = np.array([0.1, 0.2])
    mock_retrieve.return_value = (["item_1"], [0.9])
    mock_fetch_items.return_value = [{"item_id": "item_1"}]
    mock_rescore.return_value = [("item_1", 0.95)]

    valid_payload = {"user_id": "user_1", "query": "laptop", "k_results": 1}
    response = client.post("/search", json=valid_payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "laptop"
    assert len(data["results"]) == 1
    assert "relevance_scores" in data


def test_invalid_recommendation_schema():
    """Validates that Pydantic blocks invalid inputs (k_candidates < 1)."""
    invalid_payload = {"user_id": "user_100", "k_candidates": -5}
    response = client.post("/recommend", json=invalid_payload)
    
    # 422 is FastAPI's standard "Unprocessable Entity" response for validation errors
    assert response.status_code == 422