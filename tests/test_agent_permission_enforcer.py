import pytest
from src.modules.agent_permission_enforcer import AgentPermissionEnforcer

@pytest.fixture
def enforcer():
    return AgentPermissionEnforcer()

@pytest.fixture
def sample_request():
    return {
        "agent_id": "test_agent",
        "action": {
            "type": "tool_access",
            "tool_name": "search_tool",
            "operation": "read",
            "parameters": {"query": "test"}
        },
        "context": {
            "user_id": "test_user",
            "session_id": "test_session",
            "permissions": ["read", "query"]
        }
    }

def test_initialization(enforcer):
    """Test enforcer initialization."""
    assert enforcer.name == "AgentPermissionEnforcer"
    assert enforcer.priority > 0
    assert hasattr(enforcer, "permission_rules")
    assert hasattr(enforcer, "role_definitions")

def test_basic_permission_checking(enforcer, sample_request):
    """Test basic permission enforcement."""
    # Test allowed action
    result = enforcer.process(sample_request)
    assert result["enforcement_result"]["is_allowed"]
    
    # Test denied action
    sample_request["action"]["operation"] = "write"
    result = enforcer.process(sample_request)
    assert not result["enforcement_result"]["is_allowed"]

def test_role_based_access(enforcer):
    """Test role-based access control."""
    roles = {
        "reader": ["read", "query"],
        "writer": ["read", "write", "query"],
        "admin": ["read", "write", "delete", "query"]
    }
    
    for role, permissions in roles.items():
        request = {
            "agent_id": "test_agent",
            "role": role,
            "action": {
                "type": "tool_access",
                "tool_name": "document_store",
                "operation": "write"
            }
        }
        
        result = enforcer.process(request)
        assert result["enforcement_result"]["is_allowed"] == ("write" in permissions)

def test_context_based_permissions(enforcer):
    """Test context-sensitive permission enforcement."""
    contexts = [
        # Regular hours
        {"timestamp": "2025-08-28T14:00:00Z", "expected": True},
        # After hours
        {"timestamp": "2025-08-28T02:00:00Z", "expected": False},
        # Weekend
        {"timestamp": "2025-08-30T14:00:00Z", "expected": False}
    ]
    
    for ctx in contexts:
        request = sample_request.copy()
        request["context"]["timestamp"] = ctx["timestamp"]
        
        result = enforcer.process(request)
        assert result["enforcement_result"]["is_allowed"] == ctx["expected"]

def test_permission_chain_validation(enforcer, sample_request):
    """Test validation of permission chains."""
    # Test direct permission
    result = enforcer.process(sample_request)
    assert result["enforcement_result"]["is_allowed"]
    
    # Test inherited permission
    sample_request["context"]["permissions"] = ["advanced_read"]
    result = enforcer.process(sample_request)
    assert result["enforcement_result"]["is_allowed"]
    
    # Test invalid permission chain
    sample_request["context"]["permissions"] = ["invalid_permission"]
    result = enforcer.process(sample_request)
    assert not result["enforcement_result"]["is_allowed"]

def test_resource_specific_permissions(enforcer):
    """Test resource-specific permission rules."""
    resources = [
        {"name": "public_docs", "required_permission": "basic_read"},
        {"name": "private_docs", "required_permission": "advanced_read"},
        {"name": "system_docs", "required_permission": "system_access"}
    ]
    
    for resource in resources:
        request = sample_request.copy()
        request["action"]["resource"] = resource["name"]
        request["context"]["permissions"] = [resource["required_permission"]]
        
        result = enforcer.process(request)
        assert result["enforcement_result"]["is_allowed"]
        
        # Test with insufficient permission
        request["context"]["permissions"] = ["basic_read"]
        result = enforcer.process(request)
        assert result["enforcement_result"]["is_allowed"] == (
            resource["required_permission"] == "basic_read"
        )

def test_permission_escalation_prevention(enforcer, sample_request):
    """Test prevention of permission escalation attempts."""
    escalation_attempts = [
        # Try to gain admin access
        {"permissions": ["admin"], "role": "user"},
        # Try to bypass resource restrictions
        {"resource_access": "unrestricted"},
        # Try to impersonate system
        {"agent_id": "system"}
    ]
    
    for attempt in escalation_attempts:
        modified_request = sample_request.copy()
        modified_request["context"].update(attempt)
        
        result = enforcer.process(modified_request)
        assert not result["enforcement_result"]["is_allowed"]
        assert "escalation_attempt" in result["enforcement_result"]["violations"]

def test_permission_inheritance(enforcer):
    """Test permission inheritance and override rules."""
    permission_tests = [
        {
            "base_role": "user",
            "extra_permissions": ["special_read"],
            "action": "read",
            "expected": True
        },
        {
            "base_role": "user",
            "extra_permissions": [],
            "action": "admin_action",
            "expected": False
        }
    ]
    
    for test in permission_tests:
        request = {
            "agent_id": "test_agent",
            "role": test["base_role"],
            "context": {"extra_permissions": test["extra_permissions"]},
            "action": {"type": "tool_access", "operation": test["action"]}
        }
        
        result = enforcer.process(request)
        assert result["enforcement_result"]["is_allowed"] == test["expected"]

def test_temporary_permission_grants(enforcer, sample_request):
    """Test handling of temporary permission grants."""
    # Grant temporary permission
    temp_permission = {
        "permission": "special_access",
        "expires_at": "2025-08-28T15:00:00Z"
    }
    
    # Test before expiration
    sample_request["context"]["temporary_grants"] = [temp_permission]
    sample_request["context"]["current_time"] = "2025-08-28T14:00:00Z"
    
    result = enforcer.process(sample_request)
    assert "special_access" in result["enforcement_result"]["effective_permissions"]
    
    # Test after expiration
    sample_request["context"]["current_time"] = "2025-08-28T16:00:00Z"
    result = enforcer.process(sample_request)
    assert "special_access" not in result["enforcement_result"]["effective_permissions"]

def test_permission_audit_logging(enforcer, sample_request):
    """Test audit logging of permission checks."""
    # Test denied access
    sample_request["action"]["operation"] = "admin_action"
    result = enforcer.process(sample_request)
    
    assert not result["enforcement_result"]["is_allowed"]
    assert "audit_log" in result["enforcement_result"]
    assert "timestamp" in result["enforcement_result"]["audit_log"]
    assert "reason" in result["enforcement_result"]["audit_log"]
