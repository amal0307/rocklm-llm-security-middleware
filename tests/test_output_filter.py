import pytest
from src.modules.output_filter import OutputFilter, FilterResult

@pytest.fixture
def filter():
    return OutputFilter()

@pytest.fixture
def sample_data():
    return {
        "text": "This is a sample response from the AI model.",
        "context": {
            "user_id": "test_user",
            "session_id": "test_session"
        }
    }

def test_pii_detection(filter):
    """Test detection of various PII types."""
    pii_samples = {
        "email": "user@example.com",
        "phone": "123-456-7890",
        "ssn": "123-45-6789",
        "credit_card": "4111-1111-1111-1111",
        "ip_address": "192.168.1.1",
        "date_of_birth": "01/01/1990"
    }
    
    for pii_type, pii_value in pii_samples.items():
        data = {"text": f"User info: {pii_value}"}
        result = filter.process(data)
        
        assert "filter_result" in result
        assert not result["filter_result"]["is_safe"]
        assert any(e["type"] == pii_type for e in result["filter_result"]["detected_entities"])
        assert f"[REDACTED-{pii_type}]" in result["text"]

def test_policy_compliance(filter):
    """Test policy compliance checking."""
    # Test length limit
    long_text = "a" * (filter.policy_rules["max_length"] + 100)
    data = {"text": long_text}
    result = filter.process(data)
    assert not result["filter_result"]["is_safe"]
    assert any("length" in v.lower() for v in result["filter_result"]["policy_violations"])
    
    # Test restricted keywords
    for keyword in filter.policy_rules["restricted_keywords"]:
        data = {"text": f"This contains {keyword} information"}
        result = filter.process(data)
        assert not result["filter_result"]["is_safe"]
        assert any("restricted keywords" in v.lower() for v in result["filter_result"]["policy_violations"])

def test_content_toxicity(filter):
    """Test toxicity detection."""
    toxic_samples = [
        "You are stupid and worthless",
        "I hate everything about you",
        "This is the worst garbage ever"
    ]
    
    for text in toxic_samples:
        data = {"text": text}
        result = filter.process(data)
        
        assert "filter_result" in result
        assert result["filter_result"]["toxicity_score"] > 0.5
        assert not result["filter_result"]["is_safe"]

def test_safe_content_passage(filter, sample_data):
    """Test that safe content passes through."""
    result = filter.process(sample_data)
    
    assert result["filter_result"]["is_safe"]
    assert result["text"] == sample_data["text"]
    assert not result["filter_result"]["detected_entities"]
    assert not result["filter_result"]["policy_violations"]
    assert result["filter_result"]["toxicity_score"] < 0.5

def test_entity_recognition(filter):
    """Test named entity recognition."""
    text = "John Smith works at Microsoft in New York"
    data = {"text": text}
    
    result = filter.process(data)
    entities = result["filter_result"]["detected_entities"]
    
    # Should detect person, organization, and location
    entity_types = {e["type"] for e in entities}
    assert "PERSON" in entity_types
    assert "ORG" in entity_types
    assert "GPE" in entity_types

def test_redaction(filter):
    """Test redaction of sensitive information."""
    text = "Contact John at john@example.com or 123-456-7890"
    data = {"text": text}
    
    result = filter.process(data)
    redacted_text = result["text"]
    
    # Check that PII is redacted
    assert "john@example.com" not in redacted_text
    assert "123-456-7890" not in redacted_text
    assert "[REDACTED-email]" in redacted_text
    assert "[REDACTED-phone]" in redacted_text

def test_disclaimer_requirements(filter):
    """Test required disclaimer checking."""
    # Test without required disclaimers
    data = {"text": "This is a response without disclaimers"}
    result = filter.process(data)
    
    assert not result["filter_result"]["is_safe"]
    assert any("disclaimer" in v.lower() for v in result["filter_result"]["policy_violations"])
    
    # Test with all required disclaimers
    text = sample_data["text"]
    for disclaimer in filter.policy_rules["required_disclaimers"]:
        text += f"\n{disclaimer}"
    
    data = {"text": text}
    result = filter.process(data)
    
    assert not any("disclaimer" in v.lower() for v in result["filter_result"]["policy_violations"])

def test_confidence_scoring(filter):
    """Test confidence score calculation."""
    # Test with clean content
    clean_data = {"text": "This is completely safe content."}
    clean_result = filter.process(clean_data)
    assert clean_result["filter_result"]["confidence"] > 0.8
    
    # Test with multiple issues
    problematic_data = {
        "text": f"CONFIDENTIAL: Contact john@example.com. This is horrible garbage!"
    }
    prob_result = filter.process(problematic_data)
    assert prob_result["filter_result"]["confidence"] < 0.5

def test_security_logging(filter):
    """Test security event logging for unsafe content."""
    unsafe_data = {
        "text": "CONFIDENTIAL: SSN is 123-45-6789",
        "context": {
            "user_id": "test_user",
            "session_id": "test_session"
        }
    }
    
    result = filter.process(unsafe_data)
    
    assert not result["filter_result"]["is_safe"]
    # Note: In practice, would verify logs were written
