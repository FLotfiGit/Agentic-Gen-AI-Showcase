from agents.memory_enhanced import EnhancedMemory


def test_memory_record_and_recall():
    mem = EnhancedMemory()
    mem.record_action("search", "ok", tool_used="search")
    mem.remember_entity("Alpha", "Project", status="active")
    summary = mem.recall_summary()
    assert "Known entities" in summary
    assert "Event history" in summary


def test_working_memory_push_and_context():
    mem = EnhancedMemory()
    mem.working.push("Task A")
    mem.working.push("Task B")
    ctx = mem.working.get_context()
    assert "Task A" in ctx and "Task B" in ctx
