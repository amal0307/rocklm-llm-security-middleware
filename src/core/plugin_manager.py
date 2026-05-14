import importlib
import pkgutil
import inspect
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from .protocol import RockLMModule
from .config import get_config
from .logger import get_logger

class PluginManager:
    """Manages the loading and execution of security and processing modules."""
    
    def __init__(self, package: str = "src.modules"):
        self.config = get_config()
        self.logger = get_logger("plugin_manager")
        self.modules: Dict[str, RockLMModule] = {}
        self.package = package
        self.enabled = self.config.plugin.enabled
        
        if self.enabled:
            self._load_modules()
            
    def _load_modules(self) -> None:
        """Load all available modules from the specified package."""
        try:
            path = self.package.replace(".", "/")
            if not os.path.exists(path):
                self.logger.warning(f"Plugin directory {path} does not exist")
                return

            # Load internal modules
            for finder, name, ispkg in pkgutil.iter_modules([path]):
                try:
                    self._load_module(name)
                except Exception as e:
                    self.logger.error(f"Failed to load module {name}: {str(e)}")

            # Sort modules by priority
            self.modules = dict(sorted(
                self.modules.items(),
                key=lambda x: x[1].priority,
                reverse=True
            ))

            self.logger.info(f"Loaded {len(self.modules)} modules")
            
        except Exception as e:
            self.logger.error(f"Error loading modules: {str(e)}")
            raise

    def _load_module(self, name: str) -> None:
        """Load a single module and validate its interface."""
        if len(self.modules) >= self.config.plugin.max_plugins:
            self.logger.warning(f"Maximum number of plugins ({self.config.plugin.max_plugins}) reached")
            return

        mod = importlib.import_module(f"{self.package}.{name}")
        
        for attr_name, attr in inspect.getmembers(mod):
            if (isinstance(attr, type) and 
                issubclass(attr, RockLMModule) and 
                attr is not RockLMModule):
                
                instance = attr()
                
                # Validate required methods
                if not all(hasattr(instance, method) for method in ['validate_input', 'process', 'filter_output']):
                    self.logger.warning(f"Module {name} missing required methods, skipping")
                    continue
                
                self.modules[instance.name] = instance
                self.logger.debug(f"Loaded module: {instance.name}")

    def get_modules(self) -> List[RockLMModule]:
        """Get list of all loaded modules."""
        return list(self.modules.values()) if self.enabled else []

    def get_module(self, name: str) -> Optional[RockLMModule]:
        """Get a specific module by name."""
        return self.modules.get(name)

    def enable_module(self, name: str) -> bool:
        """Enable a specific module."""
        if name in self.modules:
            self.modules[name].enabled = True
            self.logger.info(f"Enabled module: {name}")
            return True
        return False

    def disable_module(self, name: str) -> bool:
        """Disable a specific module."""
        if name in self.modules:
            self.modules[name].enabled = False
            self.logger.info(f"Disabled module: {name}")
            return True
        return False

    def process_pipeline(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data through all enabled modules in priority order.
        
        Args:
            data: The data to process
            
        Returns:
            Dict[str, Any]: The processed data
        
        Raises:
            Exception: If any module blocks the processing
        """
        if not self.enabled:
            return data

        for module in self.get_modules():
            if not module.enabled:
                continue

            try:
                # Validate input
                if not module.validate_input(data):
                    raise Exception(f"Input blocked by {module.name}")

                # Process data
                data = module.process(data)

                # Filter output
                if not module.filter_output(data):
                    raise Exception(f"Output blocked by {module.name}")

            except Exception as e:
                self.logger.error(f"Error in module {module.name}: {str(e)}")
                raise

        return data
