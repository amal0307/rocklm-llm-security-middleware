import pytest
import numpy as np
from src.modules.integrity_checker import IntegrityChecker, DocumentIntegrity, IntegrityStats

@pytest.fixture
def checker():
    return IntegrityChecker()

@pytest.fixture
def sample_doc():
    return {
        "doc_id": "test_doc",
        "content": "This is a test document",
        "embedding": np.random.randn(768)  # Standard embedding size
    }

def test_document_addition(checker, sample_doc):
    """Test basic document addition and verification."""
    checker.add_document(
        sample_doc["doc_id"],
        sample_doc["content"],
        sample_doc["embedding"]
    )
    
    assert checker.verify_document(
        sample_doc["doc_id"],
        sample_doc["content"],
        sample_doc["embedding"]
    )

def test_content_tampering_detection(checker, sample_doc):
    """Test detection of content tampering."""
    checker.add_document(
        sample_doc["doc_id"],
        sample_doc["content"],
        sample_doc["embedding"]
    )
    
    # Try to verify with tampered content
    tampered_content = sample_doc["content"] + " TAMPERED"
    assert not checker.verify_document(
        sample_doc["doc_id"],
        tampered_content,
        sample_doc["embedding"]
    )

def test_embedding_tampering_detection(checker, sample_doc):
    """Test detection of embedding tampering."""
    checker.add_document(
        sample_doc["doc_id"],
        sample_doc["content"],
        sample_doc["embedding"]
    )
    
    # Try to verify with tampered embedding
    tampered_embedding = sample_doc["embedding"] + 1.0
    assert not checker.verify_document(
        sample_doc["doc_id"],
        sample_doc["content"],
        tampered_embedding
    )

def test_statistical_baseline(checker):
    """Test statistical baseline computation and anomaly detection."""
    # Add multiple normal documents
    normal_docs = []
    for i in range(100):
        doc = {
            "doc_id": f"doc_{i}",
            "content": f"Document {i}",
            "embedding": np.random.randn(768)  # Normal distribution
        }
        normal_docs.append(doc)
        checker.add_document(doc["doc_id"], doc["content"], doc["embedding"])
    
    # Add an anomalous document
    anomalous_doc = {
        "doc_id": "anomalous_doc",
        "content": "Anomalous document",
        "embedding": np.random.randn(768) * 10  # Much larger variance
    }
    
    checker.add_document(
        anomalous_doc["doc_id"],
        anomalous_doc["content"],
        anomalous_doc["embedding"]
    )
    
    # Check if anomaly is detected
    doc_info = checker.doc_registry[anomalous_doc["doc_id"]]
    assert doc_info.tampering_detected
    assert doc_info.anomaly_score > checker.anomaly_threshold

def test_integrity_scan(checker, sample_doc):
    """Test full integrity scan functionality."""
    # Add some documents
    for i in range(10):
        doc = {
            "doc_id": f"doc_{i}",
            "content": f"Document {i}",
            "embedding": np.random.randn(768)
        }
        checker.add_document(doc["doc_id"], doc["content"], doc["embedding"])
    
    # Force a scan
    scan_results = checker.run_integrity_scan(force=True)
    
    assert scan_results["total_documents"] == 10
    assert "verified_count" in scan_results
    assert "tampered_count" in scan_results
    assert "anomalous_count" in scan_results

def test_runtime_processing(checker, sample_doc):
    """Test runtime integrity checking through process method."""
    checker.add_document(
        sample_doc["doc_id"],
        sample_doc["content"],
        sample_doc["embedding"]
    )
    
    # Test with valid document
    data = {
        "document_id": sample_doc["doc_id"],
        "content": sample_doc["content"],
        "embedding": sample_doc["embedding"]
    }
    
    result = checker.process(data)
    assert result["integrity_check"]["passed"]
    
    # Test with tampered document
    data["content"] = "Tampered content"
    result = checker.process(data)
    assert not result["integrity_check"]["passed"]

def test_hmac_verification(checker):
    """Test HMAC signature verification."""
    # Generate a hash
    content = "Test content"
    hash_value = checker._compute_document_hash(content)
    
    # Sign the hash
    signature = checker._sign_hash(hash_value)
    
    # Verify valid signature
    assert checker._verify_signature(hash_value, signature)
    
    # Verify tampered signature fails
    tampered_signature = signature[:-1] + ("1" if signature[-1] != "1" else "0")
    assert not checker._verify_signature(hash_value, tampered_signature)

def test_vector_statistics(checker):
    """Test vector statistics computation and updates."""
    # Add enough documents to trigger stats computation
    embeddings = []
    for i in range(checker.min_samples):
        embedding = np.random.randn(768)
        embeddings.append(embedding)
        checker.add_document(f"doc_{i}", f"Content {i}", embedding)
    
    # Verify stats were computed
    assert checker.stats is not None
    assert isinstance(checker.stats, IntegrityStats)
    assert checker.stats.total_documents == checker.min_samples
    
    # Verify statistical properties
    assert checker.stats.mean_vector.shape == (768,)
    assert checker.stats.covariance.shape == (768, 768)
    assert isinstance(checker.stats.vector_norm_mean, float)
    assert isinstance(checker.stats.vector_norm_std, float)
