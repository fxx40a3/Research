# Project 1: Autonomous AI Agent
-----------------------------------------
## 1. Project Goal
Build a Python AI agent that can:
  1. Receive a user task.
  2. Understand and plan the task.
  3. Select and call appropriate tools.
  4. Observe tool results.
  5. Continue working until the task is complete.
  6. Return a clear final answer.

Example task:
  “Find the current weather in Tokyo and save a short report to a text file.”

## 2. Recommended Technology Stack
  **Language:** Python 3.11+
  **Model:** OpenAI, Anthropic, Gemini, or a local Ollama model
  **Environment:** `uv` or Python virtual environment
  **API framework:** FastAPI, optional
  **Interface:** Command-line interface first; Streamlit later
  **Testing:** Pytest
  **Configuration:** `.env`
  **Logging:** Python `logging`
  **Storage:** SQLite or JSON initially
  **PostgreSQL:** Use PostgreSQL for:
      Conversation history
      Users and sessions
      Agent tasks
      Tool-call logs
      Execution results
      Long-term memory
      Production deployments

  **SQLite  ** — local learning and MVP
  **PostgreSQL  ** — production application
  **PostgreSQL + pgvector  ** — AI memory and document search

## 3. Core Architecture
User->Agent Interface->Agent Controller->LLM Planner->Tool Selection->Tool Execution->Observation / Result->LLM Evaluation->Final Response

The agent operates in a loop:
  User request->Think / plan->Call a tool->Receive result->Decide whether more work is needed->Final answer

# Development Phases
## Phase 1: Project Setup
### Folder structure
ai-agent/
├── app/
│   ├── main.py
│   ├── agent.py
│   ├── llm_client.py
│   ├── prompts.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── calculator.py
│   │   ├── file_tools.py
│   │   └── web_tools.py
│   ├── memory/
│   │   └── conversation.py
│   └── models/
│       └── schemas.py
├── tests/
├── .env
├── .env.example
├── pyproject.toml
├── README.md
└── .gitignore

### Initial setup
mkdir ai-agent
cd ai-agent
powershell:
  irm https://astral.sh/uv/install.ps1 | iex
  add to path: $env:Path = "C:\Users\fxX40a3\.local\bin;$env:Path"
  uv --version
  uv init
  uv --system-certs add python-dotenv pydantic rich openai
  uv --system-certs add --dev pytest
  $env:UV_SYSTEM_CERTS = "true"
  uv add python-dotenv pydantic rich openai
  uv add --dev pytest
  uv tree

  uv run python --version
  uv run python -c "import openai, dotenv, pydantic, rich; print('All packages imported successfully')"

New-Item -ItemType Directory -Force app, app\tools, app\memory, app\models, tests
New-Item -ItemType File -Force app\__init__.py
New-Item -ItemType File -Force app\main.py
New-Item -ItemType File -Force app\agent.py
New-Item -ItemType File -Force app\llm_client.py
New-Item -ItemType File -Force app\prompts.py
New-Item -ItemType File -Force app\tools\__init__.py
New-Item -ItemType File -Force app\memory\__init__.py
New-Item -ItemType File -Force app\models\__init__.py

.gitignore
  .env
  .venv/
  __pycache__/
  *.pyc
  .pytest_cache/

New-Item -ItemType File -Force tests\test_smoke.py
uv run pytest

root: Create a `.env` file:
      OPENAI_API_KEY=your_api_key_here
Create .env.example for Git

Ollama download & setup: winget install --id Ollama.Ollama
$env:Path += ";$env:LOCALAPPDATA\Programs\Ollama"
ollama --version
ollama pull llama3.2
ollama run llama3.2

Get-ChildItem "$env:USERPROFILE\.local\bin\uv.exe" -ErrorAction SilentlyContinue
$env:Path += ";$env:USERPROFILE\.local\bin"
uv --version
uv run python -m app.main

agent:
you:
success
COMPLETED STEPS - User input → LLM client → Agent response → CLI output

NEXT STEPS - User input → Mock client → Mock response
User task
→ LLM planning
→ Tool selection
→ Tool execution
→ Tool result
→ Further planning
→ Final answer

next step: Build and test the calculator tool

Create app/tools/calculator.py
Create tests/test_calculator.py
uv run pytest
$env:Path += ";$env:USERPROFILE\.local\bin"
uv --version
uv run pytest

$env:PYTHONPATH = (Get-Location).Path
& "$env:USERPROFILE\.local\bin\uv.exe" run pytest


### DONE till now

NEXT ???


## Phase 2: Create a Basic LLM Client

The LLM client should be responsible only for communicating with the model.

Responsibilities:

- Send system instructions.
- Send conversation messages.
- Receive model responses.
- Handle API errors.
- Track token usage where available.

Do not place tool logic or application logic inside the LLM client.

---

## Phase 3: Define the Agent Contract

Create clear request and response models.

### Agent request

```text
- user_input
- conversation_id
- maximum_steps
```

### Agent response

```text
- final_answer
- steps_used
- tools_called
- success
- error_message
```

This makes the system easier to test and extend.

---

## Phase 4: Build the First Tools

Start with safe, deterministic tools.

### Tool 1: Calculator

```text
Input: mathematical expression
Output: calculated result
```

Example:

```text
calculate("25 * 4")
```

### Tool 2: Current time

```text
Input: timezone
Output: current date and time
```

### Tool 3: Text-file writer

```text
Input: filename and content
Output: success or failure
```

### Tool 4: File reader

```text
Input: filename
Output: file contents
```

Initially, avoid unrestricted shell commands or arbitrary Python execution. These can create serious security risks.

---

## Phase 5: Create a Tool Registry

The agent needs a list of available tools.

Each tool should define:

```text
- name
- description
- input schema
- execution function
```

Example concept:

```python
TOOLS = {
    "calculator": calculator,
    "read_file": read_file,
    "write_file": write_file,
}
```

The model should receive tool descriptions in a structured format so it can choose the correct tool.

---

## Phase 6: Implement the Agent Loop

The first working loop should:

1. Receive the user request.
2. Send it to the model.
3. Check whether the model requested a tool.
4. Validate the tool name.
5. Validate the tool arguments.
6. Execute the tool.
7. Send the result back to the model.
8. Repeat until a final response is produced.
9. Stop after a maximum number of steps.

Pseudo-code:

```text
messages = [system_message, user_message]

for step in range(max_steps):
    response = call_llm(messages)

    if response contains final answer:
        return final answer

    if response contains tool call:
        validate tool
        validate arguments
        result = execute tool
        add tool result to messages
        continue

return maximum-step error
```

### Important safeguards

- Set a maximum number of steps.
- Reject unknown tools.
- Validate all arguments.
- Set timeouts for external operations.
- Catch tool exceptions.
- Log every tool call.
- Never trust model-generated input automatically.

---

## Phase 7: Add Memory

Start with short-term conversation memory.

Store:

```text
- conversation ID
- user messages
- assistant responses
- tool calls
- tool results
- timestamps
```

Begin with in-memory storage or JSON. Later, migrate to SQLite.

### Memory levels

1. **Short-term memory:** Current conversation.
2. **Persistent memory:** User preferences and important facts.
3. **Knowledge memory:** Documents indexed for retrieval.

Do not add long-term memory until the basic agent loop works reliably.

---

## Phase 8: Add Error Handling

Handle at least these cases:

- Invalid API key.
- Model timeout.
- Rate limit.
- Invalid tool name.
- Invalid tool arguments.
- Tool execution error.
- Missing file.
- Maximum-step limit reached.
- Malformed model response.

The agent should return useful messages such as:

```text
I could not complete the task because the requested file was not found.
```

Avoid exposing secrets, stack traces, or internal prompts to users.

---

## Phase 9: Add a Command-Line Interface

Example usage:

```powershell
uv run python -m app.main
```

Example interaction:

```text
You: Calculate 125 * 8 and save the result to result.txt

Agent:
1. Calculated the result.
2. Saved the result to result.txt.

Final answer:
The result is 1000, and it was saved to result.txt.
```

The CLI should support:

- Interactive questions.
- Exit command.
- Conversation reset.
- Verbose logging mode.
- Maximum-step configuration.

---

## Phase 10: Add Testing

### Unit tests

Test each tool independently:

```text
- Calculator returns correct results.
- Reader handles missing files.
- Writer creates the correct content.
- Invalid arguments are rejected.
```

### Agent tests

Test:

```text
- A simple question returns a final answer.
- A calculation invokes the calculator.
- A file task invokes the correct tools.
- Unknown tools are rejected.
- Tool failures are handled.
- The agent stops at the step limit.
```

### Evaluation cases

Create a test dataset containing:

```text
- Simple questions
- Multi-step tasks
- Invalid requests
- Missing files
- Ambiguous instructions
- Tool failure scenarios
```

---

## Phase 11: Add Observability

Log:

```text
- Request ID
- User task
- Model selected
- Agent step number
- Tool name
- Tool arguments
- Tool result status
- Execution duration
- Final result
```

Do not log:

- API keys
- Passwords
- Private tokens
- Sensitive personal information

---

## Phase 12: Add a Web Interface

After the CLI works, add a simple Streamlit interface.

Features:

- Chat window.
- Conversation history.
- Tool activity display.
- Reset conversation button.
- Configuration for maximum steps.
- Error display.

The web interface should call the same `Agent` class as the CLI. Do not duplicate the agent logic.

---

## Phase 13: Add Advanced Capabilities

Only after the basic system is stable, add:

1. Web search.
2. Calendar integration.
3. Email drafting.
4. Document summarization.
5. Retrieval-augmented generation.
6. Multiple specialized agents.
7. Human approval before external actions.
8. Scheduled tasks.
9. Persistent user preferences.
10. Local LLM support with Ollama.

---

# Suggested Milestones

## Milestone 1: Basic Chat Agent

- User sends a message.
- Model returns an answer.
- CLI works.

## Milestone 2: One Tool

- Add calculator.
- Model can request and use it.
- Tool errors are handled.

## Milestone 3: Multiple Tools

- Add file reader and writer.
- Add tool registry.
- Add argument validation.

## Milestone 4: Autonomous Loop

- Support multiple tool calls.
- Add maximum-step protection.
- Add execution logs.

## Milestone 5: Memory

- Preserve conversation history.
- Add conversation IDs.
- Store data in SQLite.

## Milestone 6: Testing and Quality

- Add unit tests.
- Add integration tests.
- Add evaluation scenarios.
- Improve error handling.

## Milestone 7: User Interface

- Add Streamlit.
- Display tool activity.
- Add configuration options.

## Milestone 8: Production Preparation

- Add FastAPI.
- Add authentication.
- Add rate limits.
- Add monitoring.
- Containerize with Docker.

---

# Final MVP Requirements

The first complete version should be able to:

- Accept a natural-language task.
- Plan a multi-step operation.
- Select from at least three tools.
- Execute tools safely.
- Use tool results in later steps.
- Maintain conversation history.
- Stop safely after a configured limit.
- Handle errors clearly.
- Log important execution details.
- Pass automated tests.
- Run from both CLI and web interface.

A good first target is:

> “Build a Python CLI agent that can answer questions, perform calculations, read and write approved text files, and execute multi-step tasks using a safe tool-calling loop.”
