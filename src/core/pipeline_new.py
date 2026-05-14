import os
from typing import Dict, Any
import numpy as np
import google.genai
from google.genai.types import content_types, generation_types

from .plugin_manager import PluginManager
from .logger import get_logger
from .config import get_config
from .protocol import SecurityException

from src.modules.input_sanitizer import InputSanitizer
from src.modules.output_filter import OutputFilter
from src.modules.agent_tracker import AgentContextTracker
from src.modules.agent_permission_enforcer import AgentPermissionEnforcer
from src.modules.vector_poisoning_detector import VectorPoisoningDetector
from src.modules.retrieval_monitor import RetrievalMonitor
from src.modules.integrity_checker import IntegrityChecker
from src.modules.activation_poisoning_detector import ActivationPoisoningDetector

class Pipeline:
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("pipeline")

        # Initialize security modules in order of execution
        self.modules = []
        
        # 1. Input Validation Layer
        self.input_sanitizer = InputSanitizer()
        self.modules.append(self.input_sanitizer)
        
        # 2. Permission Layer
        self.agent_enforcer = AgentPermissionEnforcer()
        self.modules.append(self.agent_enforcer)
        
        # 3. Runtime Protection Layer
        self.agent_tracker = AgentContextTracker()
        self.activation_detector = ActivationPoisoningDetector()
        self.modules.extend([self.agent_tracker, self.activation_detector])
        
        # 4. RAG Security Layer
        self.vector_detector = VectorPoisoningDetector()
        self.retrieval_monitor = RetrievalMonitor()
        self.integrity_checker = IntegrityChecker()
        self.modules.extend([
            self.vector_detector,
            self.retrieval_monitor,
            self.integrity_checker
        ])
        
        # 5. Output Validation Layer
        self.output_filter = OutputFilter()
        self.modules.append(self.output_filter)

        # Initialize plugin system
        if self.config.plugin.enabled:
            self.plugins = PluginManager().get_modules()
            self.modules.extend(self.plugins)
        else:
            self.plugins = []

        # Sort modules by priority
        self.modules.sort(key=lambda x: x.priority)
        
        # Configure Google AI
        if not self.config.llm.api_key:
            raise ValueError("Google AI API key not configured")
            
        # Initialize with configuration
        genai.configure(
            api_key=self.config.llm.api_key,
            api_endpoint=self.config.llm.endpoint if self.config.llm.endpoint else None
        )
        
        self.logger.info(
            "Pipeline initialized with modules: %s",
            ", ".join(m.name for m in self.modules)
        )

    def run(self, user_id: str, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run the security pipeline on the input prompt.
        
        Args:
            user_id: User identifier
            prompt: Input prompt
            context: Optional context information
            
        Returns:
            Dict containing processed data and security results
        """
        # Initialize pipeline data
        data = {
            "user_id": user_id,
            "text": prompt,
            "prompt": prompt,
            "embedding": None,
            "context": context or {},
            "security_context": {
                "user_id": user_id,
                "permissions": [],
                "risk_level": 0.0
            }
        }
        
        try:
            # 1. Input Validation Layer
            self.logger.debug("Running input validation")
            data = self.input_sanitizer.process(data)
            if not data.get("sanitization_result", {}).get("is_safe", False):
                raise SecurityException("Input validation failed", data["sanitization_result"])

            # 2. Permission Layer
            self.logger.debug("Checking permissions")
            data = self.agent_enforcer.process(data)
            if not data.get("enforcement_result", {}).get("is_allowed", False):
                raise SecurityException("Permission denied", data["enforcement_result"])

            # 3. Context Tracking
            self.logger.debug("Tracking agent context")
            data = self.agent_tracker.process(data)
            if data.get("context_tracking", {}).get("is_suspicious", False):
                raise SecurityException("Context validation failed", data["context_tracking"])

            # 4. Generate embeddings for RAG security
            if self.config.security.poisoning_detection or self.config.rag.enabled:
                data["embedding"] = self._embed(data["text"])

            # 5. RAG Security Layer
            if self.config.rag.enabled:
                # Check vector poisoning
                self.logger.debug("Checking for vector poisoning")
                data = self.vector_detector.process(data)
                if data.get("poisoning_detection", {}).get("is_poisoned", False):
                    raise SecurityException("Vector poisoning detected", data["poisoning_detection"])
                
                # Monitor retrieval patterns
                self.logger.debug("Monitoring retrieval patterns")
                data = self.retrieval_monitor.process(data)
                if not data.get("retrieval_analysis", {}).get("is_safe", True):
                    raise SecurityException("Suspicious retrieval pattern", data["retrieval_analysis"])
                
                # Check knowledge base integrity
                self.logger.debug("Verifying knowledge base integrity")
                data = self.integrity_checker.process(data)
                if not data.get("integrity_check", {}).get("is_valid", False):
                    raise SecurityException("Knowledge base integrity check failed", data["integrity_check"])

            # 6. LLM Processing
            self.logger.debug("Processing with LLM")
            response = self._call_llm(data["text"])
            data["llm_response"] = response

            # 7. Activation Poisoning Detection
            self.logger.debug("Checking for activation poisoning")
            data["activation_input"] = self._extract_activation_data(data)
            data = self.activation_detector.process(data)
            if data.get("activation_check", {}).get("is_poisoned", False):
                raise SecurityException("Activation poisoning detected", data["activation_check"])

            # 8. Output Validation Layer
            self.logger.debug("Validating output")
            data["text"] = data["llm_response"]  # Set output text for filtering
            data = self.output_filter.process(data)
            if not data.get("filter_result", {}).get("is_safe", False):
                raise SecurityException("Output validation failed", data["filter_result"])

            # 9. Plugin Processing
            if self.config.plugin.enabled:
                self.logger.debug("Processing plugins")
                for plugin in self.plugins:
                    if plugin.validate_input(data):
                        data = plugin.process(data)
                        if not plugin.filter_output(data):
                            raise SecurityException(f"Plugin {plugin.name} blocked output", {})

            # Prepare final response
            return {
                "text": data["text"],
                "is_safe": True,
                "security_results": {
                    "input_validation": data.get("sanitization_result"),
                    "permissions": data.get("enforcement_result"),
                    "context_tracking": data.get("context_tracking"),
                    "vector_poisoning": data.get("poisoning_detection"),
                    "retrieval_analysis": data.get("retrieval_analysis"),
                    "integrity_check": data.get("integrity_check"),
                    "activation_check": data.get("activation_check"),
                    "output_validation": data.get("filter_result")
                }
            }

        except SecurityException as e:
            self.logger.warning(
                "Security check failed: %s", 
                e.message,
                extra={"security_data": e.security_data}
            )
            return {
                "text": None,
                "is_safe": False,
                "error": e.message,
                "security_results": e.security_data
            }
        except Exception as e:
            self.logger.error("Pipeline error: %s", str(e))
            raise

    def _call_llm(self, prompt: str) -> str:
        # Create model with safety settings and generation config
        model = genai.GenerativeModel(
            model_name=self.config.llm.model_name,
            generation_config=genai.GenerationConfig(
                temperature=self.config.llm.temperature,
                max_output_tokens=self.config.llm.max_tokens
            )
        )
        
        # Generate response
        response = model.generate_content(prompt)
        if not response.parts:
            raise ValueError("Empty response from model")
            
        return response.text

    def _embed(self, text: str):
        # Create embedding model
        model = genai.GenerativeModel(model_name=self.config.llm.embedding_model)
        
        # Generate embeddings
        result = model.embed_content(
            content={"parts": [{"text": text}]},
            task_type="retrieval_document"
        )
        
        if not result or not hasattr(result, "embedding"):
            raise ValueError("Failed to generate embedding")
            
        return result.embedding

    def _extract_activation_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data needed for activation poisoning detection."""
        return {
            "text": data["text"],
            "user_id": data["user_id"],
            "embedding": data.get("embedding"),
            "context": data.get("context", {})
        }