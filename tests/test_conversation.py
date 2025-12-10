"""Tests for conversation history management."""
import pytest
from agents.conversation import ConversationHistory, Message


def test_message_creation():
    """Test creating a message."""
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.timestamp is not None


def test_conversation_init():
    """Test conversation initialization."""
    conv = ConversationHistory(max_messages=10)
    assert len(conv) == 0
    
    # With system message
    conv_with_system = ConversationHistory(system_message="You are a helpful assistant")
    assert len(conv_with_system.messages) == 1
    assert conv_with_system.messages[0].role == "system"


def test_add_messages():
    """Test adding messages to conversation."""
    conv = ConversationHistory()
    
    conv.add_user_message("Hello")
    assert len(conv) == 1
    
    conv.add_assistant_message("Hi there!")
    assert len(conv) == 2
    
    conv.add_tool_result("calculator", "42")
    assert len(conv) == 3
    
    messages = conv.get_messages()
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"
    assert messages[2].role == "tool"


def test_context_window():
    """Test context window limiting."""
    conv = ConversationHistory(max_messages=3)
    
    for i in range(5):
        conv.add_user_message(f"Message {i}")
    
    # Should only keep last 3 messages
    assert len(conv) == 3
    messages = conv.get_messages()
    assert messages[-1].content == "Message 4"


def test_context_window_with_system():
    """Test context window preserves system message."""
    conv = ConversationHistory(max_messages=2, system_message="System prompt")
    
    conv.add_user_message("Message 1")
    conv.add_user_message("Message 2")
    conv.add_user_message("Message 3")
    
    # Should have system + last 2 messages
    assert len(conv.messages) == 3
    assert conv.messages[0].role == "system"
    assert conv.messages[-1].content == "Message 3"


def test_format_for_openai():
    """Test OpenAI API format."""
    conv = ConversationHistory(system_message="You are helpful")
    conv.add_user_message("Hello")
    conv.add_assistant_message("Hi!")
    
    formatted = conv.format_for_openai()
    assert len(formatted) == 3
    assert formatted[0]["role"] == "system"
    assert formatted[1]["role"] == "user"
    assert formatted[2]["role"] == "assistant"
    assert "content" in formatted[0]


def test_format_as_text():
    """Test text formatting."""
    conv = ConversationHistory()
    conv.add_user_message("Hello")
    conv.add_assistant_message("Hi!")
    
    text = conv.format_as_text()
    assert "USER: Hello" in text
    assert "ASSISTANT: Hi!" in text
    
    # With timestamps
    text_with_ts = conv.format_as_text(include_timestamps=True)
    assert "[" in text_with_ts  # Timestamp marker


def test_save_and_load(tmp_path):
    """Test saving and loading conversation."""
    conv = ConversationHistory(max_messages=10, system_message="Test system")
    conv.add_user_message("Hello")
    conv.add_assistant_message("Hi!")
    
    # Save
    save_path = tmp_path / "conv.json"
    conv.save(str(save_path))
    assert save_path.exists()
    
    # Load
    loaded = ConversationHistory.load(str(save_path))
    assert len(loaded.messages) == len(conv.messages)
    assert loaded.system_message == conv.system_message
    assert loaded.max_messages == conv.max_messages


def test_clear():
    """Test clearing conversation."""
    conv = ConversationHistory(system_message="System")
    conv.add_user_message("Message 1")
    conv.add_user_message("Message 2")
    
    # Clear but keep system
    conv.clear(keep_system=True)
    assert len(conv.messages) == 1
    assert conv.messages[0].role == "system"
    
    # Clear everything
    conv.add_user_message("New message")
    conv.clear(keep_system=False)
    assert len(conv.messages) == 0


def test_get_last_n_messages():
    """Test getting last N messages."""
    conv = ConversationHistory()
    for i in range(5):
        conv.add_user_message(f"Message {i}")
    
    last_2 = conv.get_last_n_messages(2)
    assert len(last_2) == 2
    assert last_2[-1].content == "Message 4"
    
    # Request more than available
    last_10 = conv.get_last_n_messages(10)
    assert len(last_10) == 5


def test_message_metadata():
    """Test message with metadata."""
    conv = ConversationHistory()
    conv.add_assistant_message("Response", metadata={"model": "gpt-4", "tokens": 42})
    
    messages = conv.get_messages()
    assert messages[0].metadata["model"] == "gpt-4"
    assert messages[0].metadata["tokens"] == 42
