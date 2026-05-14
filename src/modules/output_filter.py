import re
from typing import Dict, Any, List, Set, Optional, Tuple
from dataclasses import dataclass
import spacy
from transformers import pipeline
from ..core.protocol import RockLMModule
from ..core.config import get_config
from ..core.logger import get_logger

@dataclass
class FilterResult:
    """Results from output filtering."""
    is_safe: bool
    filtered_text: str
    detected_entities: List[Dict[str, Any]]
    policy_violations: List[str]
    toxicity_score: float
    confidence: float

class OutputFilter(RockLMModule):
    """
    Prevents output leakage by filtering sensitive information and enforcing content policies.
    
    Features:
    1. PII Detection and Redaction
    2. Entity Recognition
    3. Policy Compliance Checking
    4. Toxicity Detection
    5. Content Safety Validation
    """
    
    def __init__(self):
        super().__init__()
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Set module properties
        self.priority = 5  # High priority for output filtering
        self.name = "OutputFilter"
        
        # Initialize NLP components
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.toxicity_classifier = pipeline("text-classification", 
                                             model="unitary/toxic-bert",
                                             device=-1)  # CPU
        except Exception as e:
            self.logger.error(f"Failed to initialize NLP components: {str(e)}")
            raise
            
        # Load PII patterns
        self._load_pii_patterns()
        
        # Load policy rules
        self._load_policy_rules()
        
    def _load_pii_patterns(self) -> None:
        """Initialize regex patterns for PII detection."""
        self.pii_patterns = {
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'date_of_birth': r'\b\d{2}[-/]\d{2}[-/]\d{4}\b'
        }
        
    def _load_policy_rules(self) -> None:
        """Initialize policy compliance rules."""
        self.policy_rules = {
            'max_length': 2000,  # Prevent overly long responses
            'restricted_keywords': {
                'internal_only', 'confidential', 'proprietary',
                'classified', 'secret', 'private'
            },
            'content_blacklist': {
                'harmful_content', 'hate_speech',
                'personal_attacks', 'explicit_content',
                'promotion_of_violence', 'hate_groups'
            }
        }
        
    def _detect_pii(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect PII entities in text using regex and NER.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List[Dict[str, Any]]: Detected PII entities with type and position
        """
        detected_entities = []
        
        # Regex-based PII detection
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                detected_entities.append({
                    'type': pii_type,
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 1.0
                })
                
        # NER-based entity detection
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in {'PERSON', 'ORG', 'GPE', 'LOC'}:
                detected_entities.append({
                    'type': ent.label_,
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 0.8
                })
                
        return detected_entities
        
    def _check_policy_compliance(self, text: str) -> List[str]:
        """
        Check text for policy violations.
        
        Args:
            text: Input text to check
            
        Returns:
            List[str]: List of policy violations found
        """
        violations = []
        
        # Check text length
        if len(text) > self.policy_rules['max_length']:
            violations.append("Exceeds maximum length")
            
        # Check for restricted keywords
        words = set(text.lower().split())
        restricted_found = words.intersection(self.policy_rules['restricted_keywords'])
        if restricted_found:
            violations.append(f"Contains restricted keywords: {restricted_found}")
            
        # Check for blacklisted content
        blacklist_found = words.intersection(self.policy_rules['content_blacklist'])
        if blacklist_found:
            violations.append(f"Contains blacklisted content: {blacklist_found}")
            
        return violations
        
    def _measure_toxicity(self, text: str) -> float:
        """
        Measure content toxicity using pre-trained model.
        
        Args:
            text: Input text to analyze
            
        Returns:
            float: Toxicity score between 0 and 1
        """
        try:
            result = self.toxicity_classifier(text)
            return result[0]['score']
        except Exception as e:
            self.logger.error(f"Toxicity measurement failed: {str(e)}")
            return 0.0
            
    def _redact_pii(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """
        Redact detected PII from text.
        
        Args:
            text: Original text
            entities: List of detected PII entities
            
        Returns:
            str: Text with PII redacted
        """
        # Sort entities by start position in reverse order
        entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        # Replace each entity with redaction marker
        result = text
        for entity in entities:
            redaction = f"[REDACTED-{entity['type']}]"
            result = result[:entity['start']] + redaction + result[entity['end']:]
            
        return result
        
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data format."""
        if not isinstance(data.get("text"), str):
            self.logger.warning("Missing or invalid text in input data")
            return False
        return True
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process output text for sensitive information and policy compliance.
        
        Args:
            data: Dictionary containing:
                - text: Text to filter
                - context: Optional processing context
                
        Returns:
            Dict[str, Any]: Processed data with filtering results
        """
        text = data["text"]
        context = data.get("context", {})
        
        try:
            # Detect PII and sensitive entities
            detected_entities = self._detect_pii(text)
            
            # Check policy compliance
            policy_violations = self._check_policy_compliance(text)
            
            # Measure content toxicity
            toxicity_score = self._measure_toxicity(text)
            
            # Redact sensitive information if found
            filtered_text = self._redact_pii(text, detected_entities)
            
            # Determine if output is safe - allow educational content even with some entities
            is_safe = (
                (len(policy_violations) == 0 or all('educational_content' in v.lower() for v in policy_violations)) and
                toxicity_score < 0.7 and  # Increased threshold for educational content
                not any(v.startswith('harmful') or v.startswith('hate') for v in policy_violations)
            )
            
            # Calculate confidence score
            confidence = 1.0 - (
                0.3 * bool(detected_entities) +
                0.3 * bool(policy_violations) +
                0.4 * (toxicity_score > 0.5)
            )
            
            # Handle unsafe content with appropriate message
            if not is_safe:
                filtered_text = "I apologize, but I cannot provide information about harmful activities or generate content that could promote violence, discrimination, or harm."

            # Create filter result
            filter_result = FilterResult(
                is_safe=is_safe,
                filtered_text=filtered_text,
                detected_entities=detected_entities,
                policy_violations=policy_violations,
                toxicity_score=toxicity_score,
                confidence=confidence
            )
            
            # Add results to data
            data["filter_result"] = filter_result.__dict__
            data["text"] = filtered_text if not is_safe else text
            
            # Log unsafe content
            if not is_safe:
                security_context = self.get_security_context(data)
                self.log_security_event(
                    "unsafe_content_detected",
                    {
                        "user_id": security_context.user_id,
                        "entities": len(detected_entities),
                        "violations": policy_violations,
                        "toxicity": toxicity_score
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Error in output filtering: {str(e)}")
            raise
            
        return data
        
    def filter_output(self, data: Dict[str, Any]) -> bool:
        """Allow output if content is deemed safe."""
        filter_result = data.get("filter_result", {})
        return filter_result.get("is_safe", False)
        
    def log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log security events with detailed metrics."""
        self.logger.warning(
            f"Security Event: {event_type} - "
            f"User: {details.get('user_id', 'unknown')} - "
            f"Entities: {details.get('entities', 0)} - "
            f"Violations: {details.get('violations', [])} - "
            f"Toxicity: {details.get('toxicity', 0.0):.2f}"
        )
