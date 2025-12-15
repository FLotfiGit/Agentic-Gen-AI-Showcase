from agents.reasoning_chains import (
    ChainOfThought,
    SelfReflection,
    ReasoningChain,
    TreeOfThoughts,
    ReasoningStep,
)


def test_chain_of_thought_basic():
    cot = ChainOfThought()
    cot.observe("User wants to book a flight")
    cot.think("Need to gather departure and destination")
    cot.act("Ask for travel details")

    trace = cot.get_trace()
    assert len(trace) == 3
    assert trace[0].step_type == ReasoningStep.OBSERVE
    assert trace[1].step_type == ReasoningStep.THINK
    assert trace[2].step_type == ReasoningStep.ACT


def test_chain_of_thought_confidence():
    cot = ChainOfThought()
    cot.observe("Ambiguous request", confidence=0.6)
    cot.think("Multiple interpretations possible", confidence=0.5)
    cot.act("Request clarification", confidence=0.9)

    trace = cot.get_trace()
    assert trace[0].confidence == 0.6
    assert trace[1].confidence == 0.5
    assert trace[2].confidence == 0.9


def test_chain_of_thought_format():
    cot = ChainOfThought()
    cot.observe("Test observation")
    cot.think("Test thought")

    output = cot.format_trace()
    assert "OBSERVE" in output
    assert "THINK" in output
    assert "Test observation" in output


def test_self_reflection_confidence():
    cot = ChainOfThought()
    cot.observe("obs1", confidence=0.8)
    cot.think("think1", confidence=0.9)
    cot.act("act1", confidence=0.7)

    reflection = SelfReflection(cot)
    avg = reflection.evaluate_confidence()
    assert abs(avg - 0.8) < 0.01


def test_self_reflection_weak_steps():
    cot = ChainOfThought()
    cot.observe("obs1", confidence=0.9)
    cot.think("think1", confidence=0.5)
    cot.act("act1", confidence=0.8)

    reflection = SelfReflection(cot)
    weak = reflection.identify_weak_steps(threshold=0.7)
    assert 1 in weak
    assert 0 not in weak
    assert 2 not in weak


def test_self_reflection_add_reflection():
    cot = ChainOfThought()
    cot.observe("test")

    reflection = SelfReflection(cot)
    reflection.reflect("Need more information")
    reflection.reflect("Confidence is low")

    assert len(reflection.reflections) == 2
    assert reflection.reflections[0].content == "Need more information"


def test_reasoning_chain_success():
    chain = ReasoningChain()
    chain.add_step(lambda x: x + 1)
    chain.add_step(lambda x: x * 2)
    chain.add_step(lambda x: x - 3)

    result = chain.execute(5)
    assert result["success"] is True
    assert result["result"] == 9  # (5+1)*2-3 = 9
    assert len(result["trace"]) == 3


def test_reasoning_chain_with_validator():
    chain = ReasoningChain()
    chain.add_step(lambda x: x + 1, validator=lambda y: y > 0)
    chain.add_step(lambda x: x * 2, validator=lambda y: y < 100)

    result = chain.execute(5)
    assert result["success"] is True
    assert result["result"] == 12


def test_reasoning_chain_validation_failure():
    chain = ReasoningChain()
    chain.add_step(lambda x: x + 1)
    chain.add_step(lambda x: x * 2, validator=lambda y: y < 10)

    result = chain.execute(10)
    assert result["success"] is False
    assert "Validation failed" in result["error"]


def test_reasoning_chain_exception_handling():
    chain = ReasoningChain()
    chain.add_step(lambda x: x + 1)
    chain.add_step(lambda x: x / 0)  # Will raise exception

    result = chain.execute(5)
    assert result["success"] is False
    assert "error" in result


def test_tree_of_thoughts_branching():
    tree = TreeOfThoughts("Root problem")
    tree.branch(["Solution A", "Solution B"], scores=[0.8, 0.9])

    assert len(tree.root.children) == 2
    assert tree.root.children[0].content == "Solution A"
    assert tree.root.children[1].score == 0.9


def test_tree_of_thoughts_select_best():
    tree = TreeOfThoughts("Root")
    tree.branch(["Option 1", "Option 2", "Option 3"], scores=[0.5, 0.9, 0.7])

    best = tree.select_best_child()
    assert best is not None
    assert best.content == "Option 2"
    assert best.score == 0.9


def test_tree_of_thoughts_navigation():
    tree = TreeOfThoughts("Root")
    tree.branch(["Child A", "Child B"], scores=[0.8, 0.9])

    child = tree.select_best_child()
    tree.navigate_to(child)

    assert tree.current.content == "Child B"
    assert tree.current.parent == tree.root


def test_tree_of_thoughts_path():
    tree = TreeOfThoughts("Start")
    tree.branch(["Middle"], scores=[1.0])
    tree.navigate_to(tree.root.children[0])
    tree.branch(["End"], scores=[1.0])
    tree.navigate_to(tree.current.children[0])

    path = tree.get_path()
    assert path == ["Start", "Middle", "End"]


def test_tree_of_thoughts_format():
    tree = TreeOfThoughts("Root")
    tree.branch(["Child 1", "Child 2"], scores=[0.8, 0.9])

    output = tree.format_tree()
    assert "Root" in output
    assert "Child 1" in output
    assert "Child 2" in output
    assert "0.8" in output
