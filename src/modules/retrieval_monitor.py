import re
import numpy as np
from typing import Dict, Any
from ..core.protocol import RockLMModule
from ..core.logger import get_logger

class RetrievalMonitor(RockLMModule):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self.keyword_threshold = 5
        self.name = "RetrievalMonitor"
        self.priority = 1

    def validate_input(self, data: Dict[str, Any]) -> bool:
        """
        Validate if the input data has required fields for retrieval monitoring.
        
        Args:
            data: Input data dictionary
            
        Returns:
            bool: True if input is valid
        """
        if "text" not in data:
            self.logger.warning("Missing text field in input data")
            return False
        return True

    def filter_output(self, data: Dict[str, Any]) -> bool:
        """
        No output filtering needed for retrieval monitoring.
        
        Args:
            data: Output data dictionary
            
        Returns:
            bool: Always True as we don't filter outputs
        """
        return True

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor retrieval patterns for potential manipulation.
        
        Args:
            data: Dictionary containing:
                - text: The query text
                - embedding: Query embedding if available
                - context: Additional context
                
        Returns:
            Dict[str, Any]: Updated data with retrieval analysis results
        """
        if not self.validate_input(data):
            data["retrieval_analysis"] = {
                "is_safe": False,
                "error": "Invalid input data"
            }
            return data

        analysis = self.analyze_retrieval_patterns(
            embedding=data.get("embedding"),
            context=data.get("context", {})
        )
        
        data["retrieval_analysis"] = {
            "is_safe": analysis.get("manipulation_score", 1.0) < 0.8,
            "stats": analysis
        }
        return data

    def analyze_retrieval_patterns(self, 
                                 embedding: np.ndarray, 
                                 context: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze retrieval patterns for manipulation attempts.
        
        Args:
            embedding: Document embedding
            context: Retrieval context
            
        Returns:
            Dict[str, float]: Retrieval statistics including keyword density,
                             retrieval frequency, and priority scores
        """
        # Extract query if available
        query = context.get("query", "")
        
        # Analyze keyword density
        keywords = re.findall(r'\b\w+\b', query.lower())
        keyword_counts = {k: keywords.count(k) for k in set(keywords)}
        max_density = max(keyword_counts.values()) if keyword_counts else 0
        keyword_density = max_density / len(keywords) if keywords else 0
        
        # Calculate retrieval frequency score
        retrieval_count = context.get("retrieval_count", 1)
        time_window = context.get("time_window", 3600)  # 1 hour default
        frequency_score = retrieval_count / time_window if time_window > 0 else 0
        
        # Calculate priority based on context
        priority_factors = {
            "is_verified": context.get("is_verified", False),
            "relevance_score": context.get("relevance_score", 0.5),
            "user_trust_score": context.get("user_trust_score", 0.5)
        }
        priority_score = sum(priority_factors.values()) / len(priority_factors)
        
        retrieval_stats = {
            "keyword_density": keyword_density,
            "retrieval_frequency": frequency_score,
            "priority_score": priority_score
        }
        
        return retrieval_stats

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process retrieval request and analyze patterns.
        
        Args:
            data: Dictionary containing:
                - embedding: Document embedding
                - context: Retrieval context
                
        Returns:
            Dict[str, Any]: Processed data with retrieval analysis
        """
        embedding = data.get("embedding")
        context = data.get("context", {})
        
        # Analyze patterns
        if embedding is not None:
            retrieval_stats = self.analyze_retrieval_patterns(embedding, context)
            data["retrieval_analysis"] = retrieval_stats
            
            # Log suspicious patterns
            if retrieval_stats["keyword_density"] > 0.8 or \
               retrieval_stats["retrieval_frequency"] > 10:
                self.logger.warning(
                    "Suspicious retrieval pattern detected",
                    extra={"retrieval_stats": retrieval_stats}
                )
        
        return data
