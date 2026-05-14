from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class SecurityContext:
    """Security context passed to modules during execution."""
    user_id: str
    permissions: List[str]
    risk_level: float = 0.0
    
class SecurityException(Exception):
    """Exception raised when a security check fails."""
    def __init__(self, message: str, security_data: Dict[str, Any] = None):
        """
        Args:
            message: Error message describing the security violation
            security_data: Additional data about the security check that failed
        """
        super().__init__(message)
        self.message = message
        self.security_data = security_data or {}

class RockLMModule(ABC):
    """Base class for all RockLM security and processing modules."""
    
    def __init__(self):
        self.enabled: bool = True
        self.priority: int = 0
        self.name: str = self.__class__.__name__
        
    @abstractmethod
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """
        Validate if the input should be allowed to proceed.
        
        Args:
            data: Dictionary containing input data including:
                - text: The input text to validate
                - user_id: Unique identifier for the user
                - context: Additional context information
                - security_context: Current security state
        
        Returns:
            bool: True if input is allowed, False if it should be blocked
            
        Raises:
            SecurityException: If validation fails with security-specific details
        """
        raise NotImplementedError("Subclasses must implement validate_input()")

    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and transform the input data. Each module should add its results
        to the data dictionary under a unique key, e.g.:
            - sanitization_result
            - enforcement_result
            - tracking_result
            etc.
        
        Args:
            data: Dictionary containing:
                - text: Text to process
                - embedding: Optional text embedding if available
                - context: Additional context
                - security_context: Current security state
                - *_result: Results from previous modules
        
        Returns:
            Dict[str, Any]: Updated data dictionary with processing results
            
        Raises:
            SecurityException: If processing fails with security-specific details
        """
        raise NotImplementedError("Subclasses must implement process()")

    @abstractmethod 
    def filter_output(self, data: Dict[str, Any]) -> bool:
        """
        Validate if the processed output should be allowed.
        
        Args:
            data: Dictionary containing:
                - text: The processed output text
                - llm_response: Original LLM response if available
                - security_context: Current security state
                - *_result: Results from previous modules
                
        Returns:
            bool: True if output is allowed, False if it should be blocked
            
        Raises:
            SecurityException: If filtering fails with security-specific details
        """
        raise NotImplementedError("Subclasses must implement filter_output()")
    
    def get_security_context(self, data: Dict[str, Any]) -> SecurityContext:
        """
        Create a security context for the current operation.
        
        Args:
            data: The input data dictionary
            
        Returns:
            SecurityContext: Security context for the current operation
        """
        # Extract just the fields that SecurityContext expects
        return SecurityContext(
            user_id=data.get("user_id", "anonymous"),
            permissions=data.get("permissions", []),
            risk_level=data.get("risk_level", 0.0)
        )
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """
        Log a security-related event.
        
        Args:
            event_type: Type of security event
            details: Additional event details
        """
        # This will be implemented by specific modules as needed
        pass
