from agents.tool_registry import ToolRegistry, create_default_tools


def test_register_and_describe_tools():
    reg = ToolRegistry()
    reg.register("echo", "Echo text", lambda text: text, parameters={"text": {"type": "string"}}, required=["text"])
    desc = reg.describe_tools()
    assert "echo" in desc


def test_execute_tool_success_and_failure():
    reg = ToolRegistry()
    reg.register("add", "Add numbers", lambda a, b: a + b, parameters={"a": {}, "b": {}}, required=["a", "b"])

    ok = reg.execute_tool("add", a=2, b=3)
    assert ok.success is True
    assert ok.output == 5

    fail = reg.execute_tool("add", a=2)
    assert fail.success is False
    assert "Missing required" in fail.error


def test_default_tools_batch_execution():
    reg = create_default_tools()
    results = reg.execute_batch([
        {"name": "calculate", "params": {"expression": "2 + 2"}},
        {"name": "summarize", "params": {"text": "hello world " * 20, "max_length": 30}},
    ])
    assert len(results) == 2
    assert results[0].success is True
