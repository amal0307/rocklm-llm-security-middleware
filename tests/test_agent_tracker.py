import pytest
from src.modules.agent_tracker import AgentContextTracker

@pytest.fixture
def tracker():
    return AgentContextTracker()

@pytest.fixture
def sample_context():
    return {
        "conversation_id": "test_conv",
        "turn_id": 1,
        "current_topic": "machine learning",
        "agent_state": {
            "role": "assistant",
            "goals": ["provide information", "maintain safety"],
            "constraints": ["no harmful content"]
        },
        "chat_history": [
            {"role": "user", "content": "Tell me about AI"},
            {"role": "assistant", "content": "AI is a field of computer science..."}
        ]
    }

def test_initialization(tracker):
    """Test tracker initialization."""
    assert tracker.name == "AgentContextTracker"
    assert tracker.priority > 0
    assert hasattr(tracker, "drift_threshold")
    assert hasattr(tracker, "context_window")

def test_context_maintenance(tracker, sample_context):
    """Test basic context maintenance."""
    result = tracker.process(sample_context)
    
    assert result["tracking_result"]["is_valid"]
    assert "context_id" in result["tracking_result"]
    assert "state_hash" in result["tracking_result"]

def test_context_drift_detection(tracker, sample_context):
    """Test detection of context drift."""
    # Process initial context
    initial_result = tracker.process(sample_context)
    assert initial_result["tracking_result"]["is_valid"]
    
    # Introduce drift
    drifted_context = sample_context.copy()
    drifted_context["current_topic"] = "completely different topic"
    drifted_context["agent_state"]["goals"] = ["new goal", "another goal"]
    
    drift_result = tracker.process(drifted_context)
    assert not drift_result["tracking_result"]["is_valid"]
    assert drift_result["tracking_result"]["drift_score"] > tracker.drift_threshold

def test_conversation_coherence(tracker):
    """Test conversation coherence tracking."""
    conversation = [
        {"turn_id": 1, "content": "Tell me about AI"},
        {"turn_id": 2, "content": "AI is fascinating"},
        {"turn_id": 3, "content": "What about neural networks?"},
        {"turn_id": 4, "content": "suddenly talking about cookies"}  # Topic shift
    ]
    
    results = []
    for turn in conversation:
        context = {
            "conversation_id": "test",
            "turn_id": turn["turn_id"],
            "content": turn["content"]
        }
        results.append(tracker.process(context))
    
    assert results[0]["tracking_result"]["is_valid"]
    assert results[1]["tracking_result"]["is_valid"]
    assert results[2]["tracking_result"]["is_valid"]
    assert not results[3]["tracking_result"]["is_valid"]  # Should detect incoherence

def test_state_transition_validation(tracker, sample_context):
    """Test validation of agent state transitions."""
    # Valid state transition
    next_context = sample_context.copy()
    next_context["turn_id"] = 2
    next_context["agent_state"]["goals"].append("answer questions")
    
    result = tracker.process(next_context)
    assert result["tracking_result"]["is_valid"]
    
    # Invalid state transition (removing core constraints)
    invalid_context = next_context.copy()
    invalid_context["turn_id"] = 3
    invalid_context["agent_state"]["constraints"] = []
    
    result = tracker.process(invalid_context)
    assert not result["tracking_result"]["is_valid"]
    assert "invalid_state_transition" in result["tracking_result"]["violations"]

def test_temporal_consistency(tracker):
    """Test temporal consistency checking."""
    contexts = [
        {"turn_id": 1, "timestamp": 1000},
        {"turn_id": 2, "timestamp": 1100},
        {"turn_id": 3, "timestamp": 1200},
        {"turn_id": 2, "timestamp": 1300}  # Invalid temporal order
    ]
    
    for context in contexts[:-1]:
        result = tracker.process(context)
        assert result["tracking_result"]["is_valid"]
    
    result = tracker.process(contexts[-1])
    assert not result["tracking_result"]["is_valid"]
    assert "temporal_violation" in result["tracking_result"]["violations"]

def test_goal_consistency(tracker, sample_context):
    """Test goal consistency monitoring."""
    # Process initial goals
    initial_result = tracker.process(sample_context)
    assert initial_result["tracking_result"]["is_valid"]
    
    # Try to remove safety goal
    modified_context = sample_context.copy()
    modified_context["agent_state"]["goals"] = ["provide information"]  # Removed safety
    
    result = tracker.process(modified_context)
    assert not result["tracking_result"]["is_valid"]
    assert "goal_violation" in result["tracking_result"]["violations"]

def test_context_window_management(tracker):
    """Test management of context window."""
    # Add many turns to exceed window
    for i in range(tracker.context_window + 5):
        context = {
            "turn_id": i,
            "content": f"Turn {i}",
            "timestamp": 1000 + i
        }
        tracker.process(context)
    
    # Verify window maintenance
    state = tracker.get_tracking_state()
    assert len(state["context_history"]) <= tracker.context_window

def test_attack_detection(tracker, sample_context):
    """Test detection of potential attacks through context manipulation."""
    attack_patterns = [
        # Try to escalate privileges
        {"agent_state": {"role": "admin", "permissions": ["all"]}},
        # Try to remove constraints
        {"agent_state": {"constraints": []}},
        # Try to inject malicious goals
        {"agent_state": {"goals": ["cause harm"]}}
    ]
    
    for pattern in attack_patterns:
        modified_context = sample_context.copy()
        modified_context.update(pattern)
        
        result = tracker.process(modified_context)
        assert not result["tracking_result"]["is_valid"]
        assert "security_violation" in result["tracking_result"]["violations"]

def test_context_restoration(tracker, sample_context):
    """Test context restoration capabilities."""
    # Process initial context
    initial_result = tracker.process(sample_context)
    
    # Simulate context corruption
    corrupted_context = sample_context.copy()
    corrupted_context["agent_state"] = {}
    
    # Should detect corruption and restore
    result = tracker.process(corrupted_context)
    assert "context_restored" in result["tracking_result"]
    assert result["tracking_result"]["restored_from"] == initial_result["tracking_result"]["state_hash"]
