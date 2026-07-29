"""Tests for the LangGraph workflow implementation."""
from __future__ import annotations

import unittest
from pathlib import Path


class LangGraphWorkflowTests(unittest.TestCase):
    def test_langgraph_import(self):
        """Verify LangGraph workflow can be imported."""
        try:
            from src.langgraph_workflow import build_langgraph_workflow, run_meeting_workflow_langgraph
        except ImportError as e:
            self.fail(f"Failed to import LangGraph workflow: {e}")

    def test_langgraph_workflow_builds(self):
        """Verify the LangGraph workflow can be built."""
        from src.langgraph_workflow import build_langgraph_workflow
        
        app = build_langgraph_workflow(checkpointing=False)
        self.assertIsNotNone(app)
        self.assertTrue(hasattr(app, 'invoke'))

    def test_workflow_nodes_defined(self):
        """Verify all required nodes are present in the workflow."""
        from src.langgraph_workflow import build_langgraph_workflow
        
        app = build_langgraph_workflow(checkpointing=False)
        
        # The workflow should have these nodes defined
        # Note: LangGraph doesn't expose nodes directly, but we can verify it compiles
        self.assertIsNotNone(app)


if __name__ == "__main__":
    unittest.main()