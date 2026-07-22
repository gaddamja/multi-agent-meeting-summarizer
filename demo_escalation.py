"""Demo script to show escalation report with mocked data that meets escalation criteria."""
import json
from src.state_graph import escalation_node, final_report_node

# Mock state that simulates a completed workflow with historical data
state = {
    # Simulate topic continuity data with recurring topics
    "topic_continuity": {
        "indexed_ids": ["doc1", "doc2", "doc3", "doc4", "doc5"],
        "related": [
            {"document": "Previous meeting discussion about API migration project deadline"},
            {"document": "API migration project deadline review and blockers"},
            {"document": "Sprint planning: API migration project deadline"},
            {"document": "Retrospective: API migration challenges"},
            {"document": "Quarterly review - API migration project"},
            {"document": "Budget discussion for API migration tools"},
            {"document": "Team capacity planning for Q4"},
        ],
        "recurring_topics": [
            {"topic": "API Migration", "count": 5},
            {"topic": "Project Deadline", "count": 4},
            {"topic": "Budget Review", "count": 2},
        ],
    },
    # Simulate action history report with overdue items
    "action_history_report": {
        "meeting_source": "audio_meeting_005",
        "meeting_date": "2025-07-21",
        "participant_count": 5,
        "participants": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "total_action_items": 3,
        "status_counts": {"open": 2, "in-progress": 1, "completed": 0},
        "overdue_items": [
            {
                "id": 101,
                "action_item": "Complete API migration project deadline",
                "assignee": "Alice",
                "deadline": "2025-07-15",
                "deadline_parsed": "2025-07-15",
                "status": "open",
                "priority": "high",
            },
            {
                "id": 102,
                "action_item": "Buy groceries for team event",
                "assignee": "Bob",
                "deadline": "2025-07-20",
                "deadline_parsed": "2025-07-20",
                "status": "in-progress",
                "priority": "low",
            },
        ],
        "recurring_items": [],
        "database_path": "action_history.db",
        "reference_date": "2025-07-21",
    },
    # Add other required state fields for final_report_node
    "transcript_data": {"segments": [], "transcript": "Sample transcript"},
    "summary": {"executive_summary": "Test summary"},
    "action_items": None,
}

print("=" * 80)
print("ESCALATION DEMO - Simulated Workflow State")
print("=" * 80)

# Run escalation node
print("\n>>> Running escalation_node...")
escalation_updates = escalation_node(state)
state.update(escalation_updates)

print(f"\nEscalations detected: {len(escalation_updates['escalations']['escalations'])}")
for i, esc in enumerate(escalation_updates['escalations']['escalations'], 1):
    print(f"  {i}. {esc}")

# Run final report node
print("\n>>> Running final_report_node...")
final_updates = final_report_node(state)
state.update(final_updates)

# Display the final report
report = state["final_report"]
print("\n" + "=" * 80)
print("FINAL REPORT - Escalation Section")
print("=" * 80)
print(json.dumps(report["escalations"], indent=2, ensure_ascii=False))

print("\n" + "=" * 80)
print("Demo completed successfully!")
print("=" * 80)