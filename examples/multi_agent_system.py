"""Multi-agent system demo: collaboration, task coordination, and consensus."""

from agents.collaboration import MessageBroker, CollaborativeAgent, TaskCoordinator, ConsensusProtocol, Message
from agents.feedback_loop import FeedbackLoop, Outcome, OutcomeType


def demo_message_passing():
    """Demonstrate basic message passing between agents."""
    print("\n=== Message Passing Demo ===\n")

    broker = MessageBroker()

    # Create agents
    alice = CollaborativeAgent("Alice", broker, capabilities=["research"])
    bob = CollaborativeAgent("Bob", broker, capabilities=["coding"])
    carol = CollaborativeAgent("Carol", broker, capabilities=["testing"])

    # Set up handlers
    def handle_greeting(msg: Message):
        print(f"{msg.recipient} received greeting from {msg.sender}: {msg.content}")
        return f"Thanks, {msg.sender}!"

    alice.register_handler("greeting", handle_greeting)
    bob.register_handler("greeting", handle_greeting)
    carol.register_handler("greeting", handle_greeting)

    # Send messages
    alice.send("Bob", "Hello Bob!", "greeting")
    bob.send("Carol", "Hi Carol!", "greeting")
    carol.broadcast("Hello everyone!", "greeting")

    # Process messages
    print("\nProcessing messages:\n")
    bob.process_messages()
    carol.process_messages()
    alice.process_messages()

    print(f"\n{broker.get_message_stats()}")


def demo_task_coordination():
    """Demonstrate task coordination among agents."""
    print("\n=== Task Coordination Demo ===\n")

    broker = MessageBroker()
    coordinator = TaskCoordinator(broker)

    # Create specialized agents
    agents = [
        CollaborativeAgent("Researcher", broker, capabilities=["research"]),
        CollaborativeAgent("Developer", broker, capabilities=["coding"]),
        CollaborativeAgent("Tester", broker, capabilities=["testing"]),
    ]

    # Set up task handlers
    def handle_task(msg: Message):
        task_info = msg.content
        agent_id = msg.recipient
        print(f"{agent_id} received task: {task_info['description']}")
        return f"Task {task_info['task_id']} completed by {agent_id}"

    for agent in agents:
        agent.register_handler("task_assignment", handle_task)

    # Assign tasks
    print("Assigning tasks:\n")
    coordinator.assign_task("T1", "Research AI agents", required_capability="research")
    coordinator.assign_task("T2", "Implement feature X", required_capability="coding")
    coordinator.assign_task("T3", "Test feature X", required_capability="testing")

    print(f"\nTask Status: {coordinator.get_task_status()}")

    # Process tasks
    print("\nProcessing tasks:\n")
    for agent in agents:
        results = agent.process_messages()
        if results:
            print(f"  {agent.agent_id}: {results[0]}")

    # Complete tasks
    coordinator.complete_task("T1")
    coordinator.complete_task("T2")

    print(f"\nFinal Status: {coordinator.get_task_status()}")


def demo_consensus():
    """Demonstrate consensus protocol."""
    print("\n=== Consensus Protocol Demo ===\n")

    broker = MessageBroker()
    consensus = ConsensusProtocol(broker, threshold=0.6)

    # Create agents
    agents = [
        CollaborativeAgent(f"Agent{i}", broker)
        for i in range(5)
    ]

    # Propose decision
    print("Proposal: Should we implement new feature X?\n")
    consensus.propose("P1", "Implement new feature X", "Agent0")

    # Agents vote
    print("Voting:")
    votes = [True, True, False, True, False]  # 3/5 = 60%
    for agent, vote in zip(agents, votes):
        consensus.vote("P1", agent.agent_id, vote)
        vote_str = "YES" if vote else "NO"
        print(f"  {agent.agent_id}: {vote_str}")

    print(f"\n{consensus.get_consensus_summary('P1')}")


def demo_feedback_loop_with_agents():
    """Demonstrate feedback loop in multi-agent context."""
    print("\n=== Feedback Loop with Agents ===\n")

    # Simulate agent improving task quality through iterations
    def perform_task(state):
        quality = state.get("quality", 0.3)
        effort = state.get("effort", 1.0)
        return quality * effort

    def evaluate_task(result):
        if result >= 0.8:
            return Outcome(OutcomeType.SUCCESS, result, "High quality achieved")
        elif result >= 0.5:
            return Outcome(OutcomeType.PARTIAL, result, "Moderate quality, needs improvement")
        else:
            return Outcome(OutcomeType.FAILURE, result, "Low quality, major revision needed")

    def improve_task(state, outcome):
        improvements = {}
        if outcome.outcome_type != OutcomeType.SUCCESS:
            # Increase quality and effort
            improvements["quality"] = min(1.0, state.get("quality", 0.3) + 0.2)
            improvements["effort"] = min(2.0, state.get("effort", 1.0) + 0.3)
        return improvements

    loop = FeedbackLoop(max_iterations=5)
    result = loop.run(
        initial_state={"quality": 0.3, "effort": 1.0},
        action_fn=perform_task,
        evaluate_fn=evaluate_task,
        improve_fn=improve_task,
    )

    print("Task improvement loop:")
    print(loop.get_iteration_summary())
    print(f"\nFinal result: {result}")
    print(f"Convergence rate: {loop.get_convergence_rate():.1%}")


def main():
    demo_message_passing()
    demo_task_coordination()
    demo_consensus()
    demo_feedback_loop_with_agents()


if __name__ == "__main__":
    main()
