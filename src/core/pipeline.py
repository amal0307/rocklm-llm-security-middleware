import os
from typing import Dict, Any
import numpy as np
from google import genai 

from .plugin_manager import PluginManager
from .logger import get_logger
from .config import get_config
from .protocol import SecurityException

from src.modules.input_sanitizer import InputSanitizer
from src.modules.output_filter import OutputFilter
from src.modules.agent_tracker import AgentContextTracker
from src.modules.agent_permission_enforcer import AgentPermissionEnforcer
from src.modules.retrieval_monitor import RetrievalMonitor
from src.modules.integrity_checker import IntegrityChecker

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Missing GEMINI_API_KEY in .env")

client = genai.Client(api_key=api_key)


class Pipeline:
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("pipeline")
        self.security_logger = get_logger("security_results", security_log=True)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY in .env")
        
        self.client = genai.Client(api_key=api_key)

        # Initialize security modules in order of execution
        self.modules = []

        self.input_sanitizer = InputSanitizer()
        self.agent_enforcer = AgentPermissionEnforcer()
        self.agent_tracker = AgentContextTracker()
        self.retrieval_monitor = RetrievalMonitor()
        self.integrity_checker = IntegrityChecker()
        self.output_filter = OutputFilter()

        self.modules.extend([
            self.input_sanitizer,
            self.agent_enforcer,
            self.agent_tracker,
            self.retrieval_monitor,
            self.integrity_checker,
            self.output_filter
        ])

        if self.config.plugin.enabled:
            self.plugins = PluginManager().get_modules()
            self.modules.extend(self.plugins)
        else:
            self.plugins = []

        self.modules.sort(key=lambda x: x.priority)

        if not self.config.llm.api_key:
            raise ValueError("Google AI API key not configured")

        self.client = genai.Client(api_key=self.config.llm.api_key)

        self.logger.info(
            "Pipeline initialized with modules: %s",
            ", ".join(m.name for m in self.modules)
        )

    def run(self, user_id: str, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the security pipeline on the input prompt."""
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
            self.logger.debug("Running input validation")
            data = self.input_sanitizer.process(data)
            if not data.get("sanitization_result", {}).get("is_safe", False):
                raise SecurityException("Input validation failed", data["sanitization_result"])

            self.logger.debug("Checking permissions")
            data = self.agent_enforcer.process(data)
            if not data.get("enforcement_result", {}).get("is_allowed", False):
                raise SecurityException("Permission denied", data["enforcement_result"])

            self.logger.debug("Tracking agent context")
            data = self.agent_tracker.process(data)
            if data.get("context_tracking", {}).get("is_suspicious", False):
                raise SecurityException("Context validation failed", data["context_tracking"])

            if self.config.rag.enabled:
                self.logger.debug("Monitoring retrieval patterns")
                data = self.retrieval_monitor.process(data)
                if not data.get("retrieval_analysis", {}).get("is_safe", True):
                    raise SecurityException("Suspicious retrieval pattern", data["retrieval_analysis"])

                self.logger.debug("Verifying knowledge base integrity")
                data = self.integrity_checker.process(data)
                if not data.get("integrity_check", {}).get("is_valid", False):
                    raise SecurityException("Knowledge base integrity check failed", data["integrity_check"])

            self.logger.debug("Processing with LLM")
            response = self._call_llm(data["text"])
            data["llm_response"] = response

            self.logger.debug("Validating output")
            data["text"] = data["llm_response"]
            data = self.output_filter.process(data)
            if not data.get("filter_result", {}).get("is_safe", False):
                raise SecurityException("Output validation failed", data["filter_result"])

            if self.config.plugin.enabled:
                self.logger.debug("Processing plugins")
                for plugin in self.plugins:
                    if plugin.validate_input(data):
                        data = plugin.process(data)
                        if not plugin.filter_output(data):
                            raise SecurityException(f"Plugin {plugin.name} blocked output", {})

            # Log security results
            security_results = {
                "user_id": user_id,
                "prompt": prompt,
                "input_validation": data.get("sanitization_result"),
                "permissions": data.get("enforcement_result"),
                "context_tracking": data.get("context_tracking"),
                "retrieval_analysis": data.get("retrieval_analysis"),
                "integrity_check": data.get("integrity_check"),
                "output_validation": data.get("filter_result")
            }
            self.security_logger.info(f"Security Results: {security_results}")
            
            # Return only the filtered text
            final_text = data.get("filter_result", {}).get("filtered_text", data["text"])
            return final_text

        except SecurityException as e:
            # Log security failure with details
            self.security_logger.warning(
                f"Security check failed - User: {user_id}, Prompt: {prompt}, "
                f"Reason: {e.message}, Details: {e.security_data}"
            )
            return "I apologize, but I cannot process that request."
        except Exception as e:
            self.logger.error("Pipeline error: %s", str(e))
            raise

    def _call_llm(self, prompt: str) -> str:
        """Generate text using Google GenAI client."""
        try:
            response = self.client.models.generate_content(
                model=self.config.llm.model_name,
                contents=prompt,
                config={
                    "temperature": self.config.llm.temperature,
                    "max_output_tokens": self.config.llm.max_tokens
                }
            )
            return response.text.strip() if response and response.text else "No response."
        except Exception as e:
            self.logger.error(f"LLM generation failed: {str(e)}")
            raise

    def _embed(self, text: str) -> np.ndarray:
        """Generate embeddings for input text."""
        try:
            result = self.client.models.embed_content(
                model=self.config.llm.embedding_model,
                contents=text
            )
            return np.array(result.embeddings[0].values)
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {str(e)}")
            raise


