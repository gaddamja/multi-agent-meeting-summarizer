# Action Item Extraction Agent

The Action Item Extraction Agent identifies tasks, assignees, and deadlines from meeting transcripts using **Mistral-7B** (via Hugging Face) with **Pydantic structured output** for reliable, typed extraction.

## Features

✅ **Automatic Task Identification** - Extracts commitments, action items, and follow-ups  
✅ **Assignee Detection** - Cross-references speaker labels to assign tasks to the right person  
✅ **Deadline Parsing** - Identifies due dates and timeframes from natural language  
✅ **Priority Classification** - Categorizes tasks as high, medium, or low priority  
✅ **Context Preservation** - Includes relevant transcript quotes for each action item  
✅ **Pydantic Validation** - Structured, type-safe output with automatic validation

## Output Schema

```python
class ActionItem(BaseModel):
    action_item: str   # Task description
    assignee: str      # Responsible person (cross-referenced from speaker labels)
    deadline: str      # When it should be completed
    priority: str      # "high", "medium", or "low"
    context_quote: str # Relevant quote from transcript

class ActionItemList(BaseModel):
    action_items: List[ActionItem]  # List of extracted items
    total_count: int                # Total number of items
```

## Installation

The agent uses Hugging Face inference and Pydantic. Ensure your environment has:

```bash
pip install huggingface_hub pydantic
```

Set your Hugging Face API token:

```bash
export HF_TOKEN="your_hf_token_here"
```

Or pass it directly to the agent:

```python
agent = ActionItemAgent(hf_token="your_token")
```

## Usage

### Basic Usage with Python

```python
from src.agents.action_item_agent import ActionItemAgent
import json

# Load transcript with segments
with open("transcript.json", "r") as f:
    transcript_data = json.load(f)

# Create agent
agent = ActionItemAgent(
    model_name="Qwen/Qwen2.5-7B-Instruct",  # Default model
    hf_token="your_token"  # Or set HF_TOKEN env var
)

# Extract action items
action_items = agent.extract(
    segments=transcript_data.get("segments")
)

# Access results
print(f"Found {action_items.total_count} action items")
for item in action_items.action_items:
    print(f"- [{item.assignee}] {item.action_item}")
    print(f"  Due: {item.deadline}")
    print(f"  Priority: {item.priority}")
```

### Using the Script

```bash
# Basic extraction
python scripts/run_action_items.py transcript.json

# Save to specific file
python scripts/run_action_items.py transcript.json --output actions.json

# Use different model
python scripts/run_action_items.py transcript.json --model "meta-llama/Llama-2-7b-chat"

# Export as markdown or table
python scripts/run_action_items.py transcript.json --format markdown --output actions.md
python scripts/run_action_items.py transcript.json --format table --output actions.txt

# Provide HF token
python scripts/run_action_items.py transcript.json --hf-token "hf_xxx"
```

### Using the Orchestrator

```python
from src.orchestrator import MultiAgentOrchestrator, build_default_orchestrator

# Use the orchestrator
orch = build_default_orchestrator()

# Run just action items extraction
transcript_data = load_your_transcript()  # Load from file or other source
action_items = orch.run_action_items(
    transcript_data,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    hf_token="your_token"
)

# Run full pipeline (transcription -> summary -> action items)
result = orch.run_full_pipeline(
    audio_path="meeting.mp3",
    action_items_output="actions.json"
)

# Access all results
print(result["action_items"])  # ActionItemList dict
```

## Input Format

The agent expects transcript data with speaker-labeled segments:

```json
{
  "segments": [
    {
      "speaker": "Alice",
      "start": 0.0,
      "end": 5.5,
      "text": "I'll handle the Q3 planning document by next Friday."
    },
    {
      "speaker": "Bob",
      "start": 5.5,
      "end": 10.2,
      "text": "Great. I'll review the architecture proposal and get back by Wednesday."
    }
  ]
}
```

## How It Works

1. **Segment Normalization** - Converts transcript segments to formatted text with speaker labels
2. **Speaker Extraction** - Identifies unique speakers for assignment reference
3. **Prompt Construction** - Builds a detailed prompt that:
   - Lists all speakers
   - Includes speaker-labeling instructions
   - Provides the formatted transcript
   - Specifies exact JSON output schema
4. **Model Inference** - Sends prompt to Hugging Face model (default: Qwen/Qwen2.5-7B-Instruct)
5. **JSON Parsing** - Robustly extracts JSON from model response
6. **Pydantic Validation** - Validates and normalizes extracted items
7. **Output Return** - Returns typed `ActionItemList` with validated items

## Cross-Referencing Speaker Labels

The agent ensures assignees exactly match speaker labels from the transcript:

✅ **Correct**: Assigns to "Alice" (matches exact speaker label)  
✅ **Fallback**: Uses speaker label if name not explicitly mentioned  
⚠️ **Ambiguous Cases**: When "I" or pronouns are used, context infers the speaker  

## Supported Models

The agent works with any Hugging Face model supporting text generation:

- **Default**: `Qwen/Qwen2.5-7B-Instruct` (recommended for this task)
- **Alternatives**:
  - `meta-llama/Llama-2-7b-chat`
  - `mistralai/Mistral-7B-Instruct-v0.1`
  - `microsoft/phi-2`
  - Any other instruction-tuned model

## Configuration

### Model Selection

```python
# Use a different model
agent = ActionItemAgent(model_name="meta-llama/Llama-2-7b-chat")

# Or via script
python scripts/run_action_items.py transcript.json --model "meta-llama/Llama-2-7b-chat"
```

### Adjusting Extraction Behavior

To modify extraction behavior (e.g., strictness, priority calculation), edit the prompt in `action_item_agent.py`:

```python
# In _build_action_item_prompt():
# Adjust these instructions to be more/less strict about deadlines
"5. For deadlines, look for: dates, 'by X date', 'before', 'after', 'next week', etc.",
```

## Error Handling

```python
from src.agents.action_item_agent import ActionItemAgent

agent = ActionItemAgent()

try:
    action_items = agent.extract(segments=segments)
except ValueError as e:
    print(f"Input validation error: {e}")
except RuntimeError as e:
    print(f"Model access error: {e}")
```

Common errors:

| Error | Cause | Solution |
|-------|-------|----------|
| `HF_TOKEN not set` | Missing Hugging Face token | Set `HF_TOKEN` env var or pass `hf_token` param |
| `Repository Not Found` | Invalid model name | Verify model exists on HF and token has access |
| `Empty response` | Model returned nothing | Try different model or check prompt |
| `Failed to parse JSON` | Model output wasn't valid JSON | Reduce model temperature or try different model |

## Performance Tips

1. **Use Structured Segments** - Ensure transcript has clear `segments` with speaker labels
2. **Keep Context Relevant** - Remove off-topic sections before extraction
3. **Batch Processing** - For multiple transcripts, consider batching requests
4. **Temperature Setting** - Default is 0.3 (low) for consistent JSON output
5. **Model Choice** - Smaller models (7B) are faster; larger models (70B+) may be more accurate

## Examples

### Example 1: Simple Meeting

**Input Transcript:**
```
[Alice] 10:00: I'll prepare the budget proposal for the steering committee by end of week.
[Bob] 10:15: I'll start the infrastructure audit. Need it done by Friday EOD.
[Charlie] 10:30: Me and Bob will align on data requirements by Wednesday.
```

**output:**
```json
{
  "action_items": [
    {
      "action_item": "Prepare budget proposal for steering committee",
      "assignee": "Alice",
      "deadline": "end of week",
      "priority": "high",
      "context_quote": "I'll prepare the budget proposal for the steering committee by end of week."
    },
    {
      "action_item": "Complete infrastructure audit",
      "assignee": "Bob",
      "deadline": "Friday EOD",
      "priority": "high",
      "context_quote": "I'll start the infrastructure audit. Need it done by Friday EOD."
    },
    {
      "action_item": "Align on data requirements",
      "assignee": "Bob",
      "deadline": "Wednesday",
      "priority": "medium",
      "context_quote": "Me and Bob will align on data requirements by Wednesday."
    }
  ],
  "total_count": 3
}
```

### Example 2: Using the Orchestrator

```python
from src.orchestrator import build_default_orchestrator
import json

# Build orchestrator
orch = build_default_orchestrator()

# Load transcript
with open("meeting_transcript.json") as f:
    transcript = json.load(f)

# Extract action items
result = orch.run_action_items(
    transcript,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    hf_token="hf_xxx"
)

# Print results
for item in result.action_items:
    print(f"Task: {item.action_item}")
    print(f"Assigned to: {item.assignee}")
    print(f"Due: {item.deadline}")
    print(f"Priority: {item.priority}")
    print()
```

## Troubleshooting

### No action items extracted

- Check that segments have proper speaker labels
- Verify the transcript contains action-oriented language ("will do", "assigned to", "deadline", etc.)
- Try with a smaller model first

### Incorrect assignees

- Ensure speaker labels match exactly
- Use consistent naming conventions in transcripts
- Provide explicit names (not just pronouns)

### JSON parsing errors

- Reduce model temperature (use 0.2 or lower)
- Use a different model
- Check model output in logs

## Integration with Other Agents

The Action Item Agent integrates seamlessly with other agents:

```python
from src.orchestrator import MultiAgentOrchestrator

orch = MultiAgentOrchestrator()

# Full pipeline
result = orch.run_full_pipeline(
    audio_path="meeting.mp3",
    transcript_output="transcript.json",
    summary_output="summary.json",
    action_items_output="actions.json"
)

# Results include transcription, summary, AND action items
print("Transcription:", result["transcript"])
print("Summary:", result["summary"])
print("Action Items:", result["action_items"])
```

## Related Components

- **Summary Agent** - Creates meeting summaries (in `summary_agent.py`)
- **Transcription Agent** - Transcribes audio with speaker diarization (in `transcription_agent.py`)
- **Orchestrator** - Coordinates multiple agents (in `orchestrator.py`)

## Contributing

To improve the Action Item Agent:

1. Create test transcripts with known action items
2. Modify extraction prompts in `_build_action_item_prompt()`
3. Test with different models
4. Report issues with example transcripts

## License

Same as parent project. See LICENSE file.
