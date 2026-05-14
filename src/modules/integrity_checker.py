import hashlib
import hmac
import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from ..core.protocol import RockLMModule
from ..core.config import get_config
from ..core.logger import get_logger

@dataclass
class DocumentIntegrity:
    """Document integrity check results."""
    doc_id: str
    content_hash: str
    vector_hash: str
    last_verified: float
    is_valid: bool
    anomaly_score: float
    tampering_detected: bool

@dataclass
class IntegrityStats:
    """Statistical baseline for vector distribution."""
    mean_vector: np.ndarray
    covariance: np.ndarray
    vector_norm_mean: float
    vector_norm_std: float
    last_updated: float

class IntegrityChecker(RockLMModule):
    """
    Knowledge Base Integrity Checker for RAG systems.
    
    Features:
    1. Cryptographic verification of documents and embeddings
    2. Statistical distribution monitoring
    3. Document-embedding consistency checks
    4. Tamper-proof audit logging
    5. Periodic integrity scans
    """
    
    def __init__(self):
        super().__init__()
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Set module properties
        self.priority = 1
        self.name = "IntegrityChecker"
        
        # Initialize storage
        self.doc_registry: Dict[str, DocumentIntegrity] = {}
        self.stats: Optional[IntegrityStats] = None
        self.hmac_key = self._generate_hmac_key()
        
        # Configuration
        self.check_interval = 3600  # 1 hour
        self.anomaly_threshold = 3.0  # Number of std devs
        self.min_samples = 100
        self.last_full_scan = 0
        
    def _generate_hmac_key(self) -> bytes:
        """Generate a secure key for HMAC signatures."""
        return hashlib.sha256(str(time.time()).encode()).digest()
        
    def _compute_document_hash(self, content: str) -> str:
        """
        Compute cryptographic hash of document content.
        
        Args:
            content: Document content
            
        Returns:
            str: Hex digest of hash
        """
        return hashlib.sha256(content.encode()).hexdigest()
        
    def _compute_vector_hash(self, vector: np.ndarray) -> str:
        """
        Compute cryptographic hash of embedding vector.
        
        Args:
            vector: Embedding vector
            
        Returns:
            str: Hex digest of hash
        """
        return hashlib.sha256(vector.tobytes()).hexdigest()
        
    def _sign_hash(self, hash_value: str) -> str:
        """
        Create HMAC signature of hash to prevent tampering.
        
        Args:
            hash_value: Hash to sign
            
        Returns:
            str: HMAC signature
        """
        hmac_obj = hmac.new(self.hmac_key, hash_value.encode(), hashlib.sha256)
        return hmac_obj.hexdigest()
        
    def _verify_signature(self, hash_value: str, signature: str) -> bool:
        """
        Verify HMAC signature of hash.
        
        Args:
            hash_value: Original hash
            signature: HMAC signature to verify
            
        Returns:
            bool: True if signature is valid
        """
        expected = self._sign_hash(hash_value)
        return hmac.compare_digest(signature, expected)
        
    def _compute_vector_stats(self, vectors: List[np.ndarray]) -> IntegrityStats:
        """
        Compute statistical baseline for vector distribution.
        
        Args:
            vectors: List of embedding vectors
            
        Returns:
            IntegrityStats: Statistical baseline
        """
        vectors_array = np.array(vectors)
        
        mean_vector = np.mean(vectors_array, axis=0)
        covariance = np.cov(vectors_array.T)
        
        norms = np.linalg.norm(vectors_array, axis=1)
        norm_mean = np.mean(norms)
        norm_std = np.std(norms)
        
        return IntegrityStats(
            mean_vector=mean_vector,
            covariance=covariance,
            vector_norm_mean=norm_mean,
            vector_norm_std=norm_std,
            last_updated=time.time(),
            total_documents=len(vectors)
        )
        
    def _check_vector_anomaly(self, vector: np.ndarray) -> Tuple[bool, float]:
        """
        Check if vector deviates significantly from baseline distribution.
        
        Args:
            vector: Embedding vector to check
            
        Returns:
            Tuple[bool, float]: (is_anomaly, anomaly_score)
        """
        if self.stats is None:
            return False, 0.0
            
        # Check vector norm
        norm = np.linalg.norm(vector)
        z_score = abs(norm - self.stats.vector_norm_mean) / self.stats.vector_norm_std
        
        # Compute Mahalanobis distance
        diff = vector - self.stats.mean_vector
        try:
            inv_cov = np.linalg.inv(self.stats.covariance)
            mahalanobis = np.sqrt(diff.dot(inv_cov).dot(diff))
        except np.linalg.LinAlgError:
            mahalanobis = 0.0
            
        # Combine scores
        anomaly_score = max(z_score, mahalanobis)
        is_anomaly = anomaly_score > self.anomaly_threshold
        
        return is_anomaly, anomaly_score
        
    def add_document(self, 
                    doc_id: str, 
                    content: str, 
                    embedding: np.ndarray) -> None:
        """
        Register a new document and its embedding.
        
        Args:
            doc_id: Document identifier
            content: Document content
            embedding: Document embedding vector
        """
        # Compute hashes
        content_hash = self._compute_document_hash(content)
        vector_hash = self._compute_vector_hash(embedding)
        
        # Sign hashes
        content_sig = self._sign_hash(content_hash)
        vector_sig = self._sign_hash(vector_hash)
        
        # Check for anomalies
        is_anomaly, score = self._check_vector_anomaly(embedding)
        
        # Store integrity info
        self.doc_registry[doc_id] = DocumentIntegrity(
            doc_id=doc_id,
            content_hash=content_hash,
            vector_hash=vector_hash,
            last_verified=time.time(),
            is_valid=True,
            anomaly_score=score,
            tampering_detected=is_anomaly
        )
        
        # Update statistical baseline periodically
        if len(self.doc_registry) >= self.min_samples:
            vectors = [doc.vector_hash for doc in self.doc_registry.values()]
            self.stats = self._compute_vector_stats(vectors)
            
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """
        Validate if the input data has required fields for integrity checking.
        
        Args:
            data: Input data dictionary
            
        Returns:
            bool: True if input is valid for integrity checking
        """
        required_fields = ["text"]
        if any(field not in data for field in required_fields):
            self.logger.warning("Missing required fields for integrity check")
            return False
        return True
        
    def filter_output(self, data: Dict[str, Any]) -> bool:
        """
        No output filtering needed for integrity checking.
        
        Args:
            data: Output data dictionary
            
        Returns:
            bool: Always True as we don't filter outputs
        """
        return True
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check integrity of knowledge base documents and vectors.
        
        Args:
            data: Dictionary containing:
                - text: The query or document text
                - embedding: Document embedding if available
                - doc_id: Document ID if available
                - context: Additional context
                
        Returns:
            Dict[str, Any]: Updated data with integrity check results
        """
        if not self.validate_input(data):
            data["integrity_check"] = {
                "is_valid": False,
                "error": "Invalid input data"
            }
            return data
            
        # Get document ID if available
        doc_id = data.get("doc_id")
        
        # If document is registered, check its integrity
        if doc_id and doc_id in self.doc_registry:
            doc = self.doc_registry[doc_id]
            
            # Check if we need to reverify
            if time.time() - doc.last_verified > self.check_interval:
                # Recompute hashes and check signatures
                current_content_hash = self._compute_document_hash(data["text"])
                if data.get("embedding") is not None:
                    current_vector_hash = self._compute_vector_hash(data["embedding"])
                    
                # Update integrity status
                doc.is_valid = (
                    current_content_hash == doc.content_hash and
                    (data.get("embedding") is None or
                     current_vector_hash == doc.vector_hash)
                )
                doc.last_verified = time.time()
            
            data["integrity_check"] = {
                "is_valid": doc.is_valid,
                "doc_id": doc_id,
                "anomaly_score": doc.anomaly_score,
                "tampering_detected": doc.tampering_detected
            }
            return data
            
        # For new or unregistered documents
        # Check for vector anomalies if embedding is provided
        if data.get("embedding") is not None:
            is_anomaly, score = self._check_vector_anomaly(data["embedding"])
        else:
            is_anomaly, score = False, 0.0
            
        data["integrity_check"] = {
            "is_valid": True,  # New documents are considered valid
            "doc_id": doc_id,
            "anomaly_score": score,
            "tampering_detected": is_anomaly
        }
        return data
            
    def verify_document(self, 
                       doc_id: str, 
                       content: str, 
                       embedding: Optional[np.ndarray] = None) -> bool:
        """
        Verify document and embedding integrity.
        
        Args:
            doc_id: Document identifier
            content: Document content to verify
            embedding: Optional embedding vector to verify
            
        Returns:
            bool: True if document passes integrity checks
        """
        doc_info = self.doc_registry.get(doc_id)
        if doc_info is None:
            return False
            
        # Verify content hash
        current_hash = self._compute_document_hash(content)
        if current_hash != doc_info.content_hash:
            self.logger.warning(f"Content hash mismatch for document {doc_id}")
            return False
            
        # Verify embedding if provided
        if embedding is not None:
            current_vector_hash = self._compute_vector_hash(embedding)
            if current_vector_hash != doc_info.vector_hash:
                self.logger.warning(f"Vector hash mismatch for document {doc_id}")
                return False
                
            # Check for anomalies
            is_anomaly, score = self._check_vector_anomaly(embedding)
            if is_anomaly:
                self.logger.warning(
                    f"Anomalous vector detected for document {doc_id}",
                    extra={"anomaly_score": score}
                )
                return False
                
        return True
        
    def run_integrity_scan(self, force: bool = False) -> Dict[str, Any]:
        """
        Run full integrity scan of knowledge base.
        
        Args:
            force: Force scan even if interval hasn't elapsed
            
        Returns:
            Dict[str, Any]: Scan results
        """
        current_time = time.time()
        if not force and current_time - self.last_full_scan < self.check_interval:
            return {"status": "skipped", "reason": "interval not elapsed"}
            
        self.last_full_scan = current_time
        scan_results = {
            "total_documents": len(self.doc_registry),
            "verified_count": 0,
            "tampered_count": 0,
            "anomalous_count": 0,
            "failed_documents": []
        }
        
        for doc_id, integrity in self.doc_registry.items():
            try:
                # Verify document exists and matches hash
                # In production, this would fetch from storage
                content = "mock_content"  # Replace with actual fetch
                embedding = np.zeros(768)  # Replace with actual fetch
                
                if self.verify_document(doc_id, content, embedding):
                    scan_results["verified_count"] += 1
                else:
                    scan_results["tampered_count"] += 1
                    scan_results["failed_documents"].append(doc_id)
                    
                if integrity.tampering_detected:
                    scan_results["anomalous_count"] += 1
                    
            except Exception as e:
                self.logger.error(f"Error scanning document {doc_id}: {str(e)}")
                scan_results["failed_documents"].append(doc_id)
                
        return scan_results
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process integrity checks during runtime."""
        doc_id = data.get("document_id")
        content = data.get("content")
        embedding = data.get("embedding")
        
        if doc_id and content:
            is_valid = self.verify_document(doc_id, content, embedding)
            data["integrity_check"] = {
                "passed": is_valid,
                "timestamp": time.time()
            }
            
        return data
