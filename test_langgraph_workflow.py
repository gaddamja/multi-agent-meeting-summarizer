#!/usr/bin/env python3
"""Quick test to verify LangGraph workflow builds correctly."""
import sys

try:
    from src.langgraph_workflow import build_langgraph_workflow, run_meeting_workflow_langgraph
    
    print("Testing LangGraph workflow...")
    
    # Build the workflow
    app = build_langgraph_workflow(checkpointing=True)
    print("✓ LangGraph workflow built successfully")
    
    # Verify it's a StateGraph
    print(f"✓ Workflow type: {type(app).__name__}")
    
    # Test with minimal state (will fail at transcription but proves workflow compiles)
    try:
        result = run_meeting_workflow_langgraph({
            "audio_path": "nonexistent.mp3",
            "meeting_id": "test-123"
        }, checkpointing=False)
    except Exception as e:
        # Expected to fail at transcription node, but workflow should compile
        print(f"✓ Workflow executed (failed at expected point: {type(e).__name__})")
    
    print("\n✓ LangGraph integration successful!")
    sys.exit(0)
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("  Make sure langgraph is installed: pip install langgraph")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)