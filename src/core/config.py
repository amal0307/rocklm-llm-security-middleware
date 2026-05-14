import os
import yaml
import json
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from pathlib import Path

# Load environment variables from .env file if available
env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key.strip()] = value.strip()

@dataclass
class LLMConfig:
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.7
    max_tokens: int = 2048
    api_key: Optional[str] = None
    endpoint: Optional[str] = None 
    embedding_model: str = "models/text-embedding-004"  

@dataclass
class SecurityConfig:
    # Input/Output validation
    input_validation: bool = True
    output_filtering: bool = True
    toxicity_model: str = "toxicity-classifier"
    
    # Agent security
    permission_enforcement: bool = True
    agent_context_tracking: bool = True
    max_context_drift: float = 0.3
    
    # General security
    integrity_checks: bool = True
    max_retries: int = 3
    timeout_seconds: int = 30

    def __post_init__(self):
        pass  # No initialization needed

@dataclass
class RAGConfig:
    enabled: bool = False
    vector_store_path: str = "data/vector_store"
    similarity_threshold: float = 0.75
    max_chunks: int = 5
    chunk_size: int = 1000
    retrieval_monitoring: bool = True
    knowledge_base_integrity: bool = True
    max_retrieval_attempts: int = 3

@dataclass
class PluginConfig:
    enabled: bool = True
    plugin_dir: str = "src.modules"
    allow_external_plugins: bool = False
    plugin_timeout: int = 10
    max_plugins: int = 50

@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    log_level: str = "INFO"
    enable_audit_trail: bool = True
    max_log_size_mb: int = 100
    backup_count: int = 5
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console_logging: bool = True

class Config:
    def __init__(self, path: str = "config.yml"):
        self._data: Dict[str, Any] = {}
        self.base_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        # Load from config file if available
        if os.path.exists(path):
            with open(path, "r") as f:
                self._data = yaml.safe_load(f) or {}

        # Initialize configurations
        self.llm = self._init_llm_config()
        self.security = self._init_security_config()
        self.rag = self._init_rag_config()
        self.plugin = self._init_plugin_config()
        self.logging = self._init_logging_config()

    def _init_llm_config(self) -> LLMConfig:
        return LLMConfig(
            model_name=os.getenv("LLM_MODEL_NAME", self._data.get("model_name", "gemini-2.0-flash")),
            temperature=float(os.getenv("LLM_TEMPERATURE", self._data.get("temperature", 0.7))),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", self._data.get("max_tokens", 2048))),
            api_key=os.getenv("GEMINI_API_KEY", self._data.get("gemini_api_key")),
            endpoint=os.getenv("LLM_ENDPOINT", self._data.get("endpoint")),
            embedding_model=os.getenv("EMBEDDING_MODEL", self._data.get("embedding_model", "embedding-001"))
        )

    def _init_security_config(self) -> SecurityConfig:
        return SecurityConfig(
            toxicity_model=os.getenv("TOXICITY_MODEL", self._data.get("toxicity_model", "toxicity-classifier")),
            agent_context_tracking=bool(os.getenv("AGENT_TRACKING", self._data.get("agent_context_tracking", True))),
            max_context_drift=float(os.getenv("MAX_CONTEXT_DRIFT", self._data.get("max_context_drift", 0.3)))
        )

    def _init_rag_config(self) -> RAGConfig:
        return RAGConfig(
            vector_store_path=str(self.base_dir / "data" / "vector_store")
        )

    def _init_plugin_config(self) -> PluginConfig:
        return PluginConfig()

    def _init_logging_config(self) -> LoggingConfig:
        return LoggingConfig(
            log_dir=str(self.base_dir / "logs")
        )

    def _load_array(self, env_key: str, default) -> List[float]:
        val = os.getenv(env_key)
        if val:
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                print(f"[WARN] Failed to parse {env_key} as JSON.")
        return default

    def __getitem__(self, key: str) -> Any:
        return self._data.get(key)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "llm": {
                "model_name": self.llm.model_name,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "endpoint": self.llm.endpoint,
                "embedding_model": self.llm.embedding_model
            },
            "security": {
                "input_validation": self.security.input_validation,
                "output_filtering": self.security.output_filtering,
                "toxicity_model": self.security.toxicity_model,
                "permission_enforcement": self.security.permission_enforcement,
                "integrity_checks": self.security.integrity_checks,
                "max_retries": self.security.max_retries,
                "timeout_seconds": self.security.timeout_seconds
            },
            "rag": {
                "vector_store_path": self.rag.vector_store_path,
                "similarity_threshold": self.rag.similarity_threshold,
                "max_chunks": self.rag.max_chunks,
                "chunk_size": self.rag.chunk_size,
                "retrieval_monitoring": self.rag.retrieval_monitoring,
                "knowledge_base_integrity": self.rag.knowledge_base_integrity,
                "max_retrieval_attempts": self.rag.max_retrieval_attempts
            },
            "plugin": {
                "enabled": self.plugin.enabled,
                "plugin_dir": self.plugin.plugin_dir,
                "allow_external_plugins": self.plugin.allow_external_plugins,
                "plugin_timeout": self.plugin.plugin_timeout,
                "max_plugins": self.plugin.max_plugins
            },
            "logging": {
                "log_dir": self.logging.log_dir,
                "log_level": self.logging.log_level,
                "enable_audit_trail": self.logging.enable_audit_trail,
                "max_log_size_mb": self.logging.max_log_size_mb,
                "backup_count": self.logging.backup_count,
                "log_format": self.logging.log_format,
                "console_logging": self.logging.console_logging
            }
        }

def get_config() -> Config:
    return Config()
