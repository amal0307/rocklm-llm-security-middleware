import pytest
import numpy as np
from src.modules.retrieval_monitor import RetrievalMonitor

@pytest.fixture
def monitor():
    return RetrievalMonitor()

@pytest.fixture
def sample_data():
    return {
        "embedding": np.random.randn(768),
        "context": {
            "query": "test query about machine learning",
            "is_verified": True,
            "relevance_score": 0.8,
            "user_trust_score": 0.9,
            "retrieval_count": 1,
            "time_window": 3600
        }
    }

def test_normal_retrieval_patterns(monitor, sample_data):
    """Test normal retrieval pattern analysis."""
    result = monitor.process(sample_data)
    
    assert "retrieval_analysis" in result
    analysis = result["retrieval_analysis"]
    
    assert "keyword_density" in analysis
    assert "retrieval_frequency" in analysis
    assert "priority_score" in analysis
    
    # Normal patterns should have reasonable scores
    assert 0 <= analysis["keyword_density"] <= 1
    assert analysis["retrieval_frequency"] >= 0
    assert 0 <= analysis["priority_score"] <= 1

def test_keyword_flooding_detection(monitor):
    """Test detection of keyword flooding attacks."""
    # Create data with repeated keywords
    flood_data = {
        "embedding": np.random.randn(768),
        "context": {
            "query": "password password password password password password",
            "retrieval_count": 1,
            "time_window": 3600
        }
    }
    
    result = monitor.process(flood_data)
    analysis = result["retrieval_analysis"]
    
    # High keyword density should be detected
    assert analysis["keyword_density"] > 0.8

def test_high_frequency_detection(monitor, sample_data):
    """Test detection of high-frequency retrievals."""
    # Simulate rapid retrievals
    sample_data["context"]["retrieval_count"] = 100
    sample_data["context"]["time_window"] = 60  # 1 minute
    
    result = monitor.process(sample_data)
    analysis = result["retrieval_analysis"]
    
    # High frequency should be reflected in score
    assert analysis["retrieval_frequency"] > 1.0

def test_priority_scoring(monitor, sample_data):
    """Test priority scoring based on context factors."""
    # Test with varying trust factors
    test_cases = [
        # is_verified, relevance_score, user_trust_score, expected_priority
        (True, 0.9, 0.9, 0.9),  # High trust
        (False, 0.5, 0.5, 0.5),  # Medium trust
        (False, 0.2, 0.2, 0.2)   # Low trust
    ]
    
    for verified, relevance, trust, expected in test_cases:
        sample_data["context"]["is_verified"] = verified
        sample_data["context"]["relevance_score"] = relevance
        sample_data["context"]["user_trust_score"] = trust
        
        result = monitor.process(sample_data)
        analysis = result["retrieval_analysis"]
        
        assert abs(analysis["priority_score"] - expected) < 0.2

def test_missing_context_handling(monitor):
    """Test handling of missing context information."""
    minimal_data = {
        "embedding": np.random.randn(768),
        "context": {
            "query": "simple query"
        }
    }
    
    result = monitor.process(minimal_data)
    analysis = result["retrieval_analysis"]
    
    # Should handle missing context gracefully
    assert "keyword_density" in analysis
    assert "retrieval_frequency" in analysis
    assert "priority_score" in analysis

def test_malformed_queries(monitor):
    """Test handling of malformed or suspicious queries."""
    suspicious_queries = [
        "SELECT * FROM users",  # SQL injection attempt
        "../../etc/passwd",     # Path traversal attempt
        "<script>alert(1)</script>",  # XSS attempt
        "a" * 1000             # Very long query
    ]
    
    for query in suspicious_queries:
        data = {
            "embedding": np.random.randn(768),
            "context": {"query": query}
        }
        
        result = monitor.process(data)
        analysis = result["retrieval_analysis"]
        
        # Should detect suspicious patterns
        assert analysis["keyword_density"] > 0.5 or \
               analysis["priority_score"] < 0.5

def test_multi_query_monitoring(monitor):
    """Test monitoring of multiple sequential queries."""
    queries = [
        "machine learning basics",
        "advanced deep learning",
        "neural networks training",
        "machine machine machine machine"  # Suspicious
    ]
    
    results = []
    for query in queries:
        data = {
            "embedding": np.random.randn(768),
            "context": {
                "query": query,
                "retrieval_count": 1,
                "time_window": 3600
            }
        }
        result = monitor.process(data)
        results.append(result["retrieval_analysis"])
    
    # First three should have normal patterns
    for i in range(3):
        assert results[i]["keyword_density"] < 0.5
    
    # Last one should show suspicious pattern
    assert results[-1]["keyword_density"] > 0.5

def test_logging_behavior(monitor, sample_data):
    """Test proper logging of suspicious patterns."""
    # Create suspicious pattern
    sample_data["context"]["query"] = "attack " * 10
    sample_data["context"]["retrieval_count"] = 1000
    sample_data["context"]["time_window"] = 60
    
    result = monitor.process(sample_data)
    analysis = result["retrieval_analysis"]
    
    # Verify high-risk patterns are detected
    assert analysis["keyword_density"] > 0.8 or \
           analysis["retrieval_frequency"] > 10
