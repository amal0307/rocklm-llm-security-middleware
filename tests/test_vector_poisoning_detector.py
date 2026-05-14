import pytest
import numpy as np
from src.modules.vector_poisoning_detector import VectorPoisoningDetector, EmbeddingStats, DetectionResult

@pytest.fixture
def detector():
    detector = VectorPoisoningDetector()
    # Initialize with some safe embeddings
    safe_embeddings = [np.random.randn(768) for _ in range(10)]
    detector._initialize_safe_distribution()
    return detector

@pytest.fixture
def sample_data():
    return {
        "embedding": np.random.randn(768),
        "document_id": "test_doc",
        "context": {
            "query": "test query",
            "retrieval_count": 1,
            "time_window": 3600
        }
    }

def test_initialization(detector):
    """Test detector initialization and safe distribution."""
    assert detector.stats is not None
    assert detector.sensitivity == 2.5
    assert detector.min_safe_samples == 10
    assert len(detector.recent_embeddings) == 0
    assert isinstance(detector._safe_document_ids, set)

def test_cholesky_decomposition(detector):
    """Test Cholesky decomposition computation."""
    embeddings = np.random.randn(20, 768)
    cholesky_L = detector._compute_cholesky_factor(embeddings)
    
    # Verify Cholesky properties
    assert cholesky_L.shape == (768, 768)
    # Lower triangular test
    assert np.allclose(cholesky_L, np.tril(cholesky_L))
    
def test_mahalanobis_distance(detector):
    """Test Mahalanobis distance computation."""
    # Generate test data
    mean = np.random.randn(768)
    embeddings = np.random.randn(20, 768)
    cholesky_L = detector._compute_cholesky_factor(embeddings)
    
    # Test vector
    x = np.random.randn(768)
    
    # Compute distance
    dist = detector._compute_mahalanobis_distance(x, mean, cholesky_L)
    
    assert isinstance(dist, float)
    assert dist >= 0  # Distance should be non-negative

def test_clean_embedding_processing(detector, sample_data):
    """Test processing of normal, clean embeddings."""
    # Process clean data
    result = detector.process(sample_data)
    
    assert "poisoning_detection" in result
    detection = result["poisoning_detection"]
    assert not detection["is_poisoned"]
    assert 0 <= detection["similarity_score"] <= 1
    assert "mahalanobis_distance" in detection

def test_poisoned_embedding_detection(detector, sample_data):
    """Test detection of poisoned embeddings."""
    # Create anomalous embedding
    sample_data["embedding"] = np.random.randn(768) * 10  # Much larger variance
    
    # Process potentially poisoned data
    result = detector.process(sample_data)
    
    assert "poisoning_detection" in result
    detection = result["poisoning_detection"]
    assert detection["is_poisoned"]
    assert detection["mahalanobis_distance"] > detector.sensitivity * np.sqrt(768)

def test_distribution_updates(detector):
    """Test updating of distribution statistics."""
    # Add several normal embeddings
    for i in range(20):
        data = {
            "embedding": np.random.randn(768),
            "document_id": f"doc_{i}",
            "context": {}
        }
        detector.process(data)
    
    # Verify stats were updated
    assert detector.stats is not None
    assert len(detector.recent_embeddings) > 0
    assert detector.stats.mean_vector.shape == (768,)

def test_adaptive_thresholding(detector, sample_data):
    """Test adaptive threshold based on embedding dimensionality."""
    # Process with different embedding dimensions
    dims = [128, 256, 512, 768]
    
    for dim in dims:
        data = sample_data.copy()
        data["embedding"] = np.random.randn(dim) * 5  # Potentially anomalous
        
        result = detector.process(data)
        detection = result["poisoning_detection"]
        
        # Verify threshold scales with dimensionality
        threshold = detector.sensitivity * np.sqrt(dim)
        assert detection["is_poisoned"] == (detection["mahalanobis_distance"] > threshold)

def test_retrieval_integration(detector, sample_data):
    """Test integration with retrieval monitoring."""
    # Add retrieval context
    sample_data["context"]["retrieval_analysis"] = {
        "keyword_density": 0.5,
        "retrieval_frequency": 1.0,
        "priority_score": 0.8
    }
    
    result = detector.process(sample_data)
    
    assert "poisoning_detection" in result
    detection = result["poisoning_detection"]
    assert "retrieval_stats" in detection

def test_security_event_logging(detector, sample_data):
    """Test security event logging for poisoning detection."""
    # Create suspicious embedding
    sample_data["embedding"] = np.random.randn(768) * 10
    
    # Process with security context
    sample_data["security_context"] = {
        "user_id": "test_user",
        "session_id": "test_session"
    }
    
    result = detector.process(sample_data)
    
    # Verify detection and logging occurred
    assert result["poisoning_detection"]["is_poisoned"]
    # Note: In practice, would verify logs were written

def test_output_filtering(detector, sample_data):
    """Test output filtering based on detection results."""
    # Test with clean embedding
    result = detector.process(sample_data.copy())
    assert detector.filter_output(result)
    
    # Test with poisoned embedding
    sample_data["embedding"] = np.random.randn(768) * 10
    result = detector.process(sample_data)
    assert not detector.filter_output(result)
