import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from src.agents.action_item_agent import ActionItem, ActionItemList
from src.state_graph import StateGraph, StateNode, run_meeting_workflow


class StateGraphTests(unittest.TestCase):
    def test_nodes_return_partial_state_updates(self):
        graph = StateGraph(
            [
                StateNode("first", "set a", lambda state: {"a": 1}, "last"),
                StateNode("last", "derive b", lambda state: {"b": state["a"] + 1}),
            ],
            start_state="first",
            final_state="last",
        )

        result = graph.run({"input": "preserved"})

        self.assertEqual(result["input"], "preserved")
        self.assertEqual(result["a"], 1)
        self.assertEqual(result["b"], 2)
        self.assertEqual(result["current_node"], "last")

    def test_workflow_nodes_invoke_agents_directly(self):
        summary = {
            "executive_summary": "Mike owns the report.",
            "discussion_topics": [{"topic": "Reporting"}],
        }
        actions = ActionItemList(
            action_items=[
                ActionItem(
                    action_item="Send report",
                    assignee="Mike",
                    deadline="2026-07-15",
                    priority="high",
                    context_quote="Mike will send it.",
                )
            ],
            total_count=1,
        )

        summary_agent = MagicMock()
        summary_agent.summarize.return_value = summary
        action_agent = MagicMock()
        action_agent.extract.return_value = actions
        history_agent = MagicMock()
        history_agent.save_action_items.return_value = [42]
        history_agent.build_health_report.return_value = {"overdue_items": []}
        topic_agent = MagicMock()
        topic_agent.index_meeting.return_value = ["meeting-summary"]
        topic_agent.query_related.return_value = []
        topic_agent.find_recurring_topics.return_value = []
        topic_module = types.SimpleNamespace(TopicContinuityAgent=MagicMock(return_value=topic_agent))

        with (
            patch("src.state_graph.SummaryAgent", return_value=summary_agent) as summary_class,
            patch("src.state_graph.ActionItemAgent", return_value=action_agent) as action_class,
            patch("src.state_graph.ActionHistoryAgent", return_value=history_agent) as history_class,
            patch.dict(sys.modules, {"src.agents.topic_continuity_agent": topic_module}),
        ):
            result = run_meeting_workflow(
                {
                    "transcript_data": {"transcript": "Mike will send it.", "segments": []},
                    "meeting_source": "weekly-sync",
                    "meeting_date": "2026-07-14",
                }
            )

        summary_class.assert_called_once()
        action_class.assert_called_once()
        history_class.assert_called_once()
        summary_agent.summarize.assert_called_once()
        action_agent.extract.assert_called_once()
        history_agent.save_action_items.assert_called_once()
        topic_agent.index_meeting.assert_called_once()
        self.assertEqual(result["saved_action_item_ids"], [42])
        self.assertEqual(result["final_report"]["action_items"]["total_count"], 1)


if __name__ == "__main__":
    unittest.main()
