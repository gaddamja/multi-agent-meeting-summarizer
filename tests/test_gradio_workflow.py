import unittest
from unittest.mock import patch

from scripts.gradio_app import process_meeting_with_error_handling
from src.agents.action_item_agent import ActionItemList


class GradioWorkflowTests(unittest.TestCase):
    def test_manual_transcript_uses_state_graph_workflow(self):
        workflow_result = {
            "transcript_data": {"transcript": "A short meeting.", "segments": []},
            "summary": {"executive_summary": "A short meeting."},
            "action_items": ActionItemList(action_items=[], total_count=0),
            "saved_action_item_ids": [],
            "action_history_report": {"recurring_items": []},
        }

        with (
            patch("scripts.gradio_app.run_meeting_workflow", return_value=workflow_result) as run,
            patch("scripts.gradio_app._build_topic_graph_image", return_value=""),
            patch("scripts.gradio_app._build_pdf_report"),
        ):
            result = process_meeting_with_error_handling(
                None,
                "A short meeting.",
                "",
                "small",
                "Qwen/Qwen2.5-7B-Instruct",
                "Qwen/Qwen2.5-7B-Instruct",
                "manual-test",
            )

        run.assert_called_once()
        self.assertEqual(len(result), 10)
        self.assertEqual(result[1], "A short meeting.")
        self.assertEqual(result[8]["executive_summary"], "A short meeting.")


if __name__ == "__main__":
    unittest.main()
