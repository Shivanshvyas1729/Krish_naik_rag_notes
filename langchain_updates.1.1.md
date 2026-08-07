# LangChain v1.1 & LangGraph Agent Architecture (Comprehensive Study Notes)

---

## 📋 Overview & Module Index

This document provides a comprehensive theoretical breakdown, architecture diagrams, and production-grade code walkthroughs for **LangChain v1.1 / 1.x** built on the **LangGraph** execution engine.

| Notebook | Topic & Core Focus | Key LangChain v1.1 APIs Used |
| :--- | :--- | :--- |
| **`1-langchainintro.ipynb`** | Agent Foundations & Graph Engine | `create_agent()`, `@tool`, `agent.invoke()` |
| **`2-modelintegration.ipynb`** | Universal Model Integration, Streaming & Batching | `init_chat_model()`, `ChatOpenAI()`, `ChatGroq()`, `ChatGoogleGenerativeAI()`, `model.stream()`, `model.batch()` |
| **`3-tools.ipynb`** | Tool Definition, Schemas & Both Binding Methods | `@tool`, `model.bind_tools()`, `ai_msg.tool_calls`, `ToolMessage` |
| **`4-messages.ipynb`** | Canonical Message Schema & Token Tracking | `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`, `usage_metadata` |
| **`5-structuredoutput.ipynb`** | Enforced Schema Parsing & Validation | `with_structured_output()`, `response_format`, `Pydantic`, `TypedDict`, `@dataclass` |
| **`6-middleware.ipynb`** | Agent Middleware, Memory & Human-in-the-Loop | `SummarizationMiddleware`, `HumanInTheLoopMiddleware`, `InMemorySaver`, `Command` |

---

## 📖 Module Deep Dives & Theoretical Foundations

### 1. `1-langchainintro.ipynb` – Agent Foundations & High-Level Architecture

#### 🧠 Theory & Core Concepts
An **AI Agent** uses a Large Language Model (LLM) as a central reasoning engine to dynamically decide which tools to call, what parameters to extract, and how to sequence actions to satisfy a user request.

- **Legacy vs. LangChain v1.1 Agent Architecture**:
  - *Legacy (`AgentExecutor`)*: Relied on complex Python loops and manual memory state passing.
  - *LangChain v1.1 (`create_agent`)*: Constructs a stateful, compiled graph engine powered by **LangGraph** under the hood. It natively handles cyclic agent loops, message state persistence, and tool execution error recovery.

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_weather],
    system_prompt="You are a helpful assistant."
)

# Invocation accepts messages in OpenAI or LangChain format
response = agent.invoke({"messages": [{"role": "user", "content": "What is the weather in New York?"}]})
print(response["messages"])
```

---

### 2. `2-modelintegration.ipynb` – Universal Model Provider Loading, Streaming & Batching

#### 🧠 Theory & Core Concepts
LangChain v1.1 decouples provider-specific code from application logic using a universal model initializer and standardized execution paradigms.

1. **Universal Model Initialization (`init_chat_model`)**:
   Instead of importing provider classes directly (`ChatOpenAI`, `ChatGroq`, `ChatGoogleGenerativeAI`), `init_chat_model()` instantiates any LLM via string identifiers, making provider migration seamless.

   ```python
   from langchain.chat_models import init_chat_model

   model_openai = init_chat_model("gpt-4o-mini")
   model_groq = init_chat_model("groq:llama-3.3-70b-versatile")
   model_gemini = init_chat_model("google_genai:gemini-1.5-flash")
   ```

2. **Streaming Output (`model.stream()`)**:
   - *Concept*: LLMs generate text token by token. Calling `model.stream()` uses HTTP chunked transfer encoding to yield `AIMessageChunk` objects in real time.
   - *UX Benefit*: Eliminates user waiting time by displaying output progressively (reduces Time-To-First-Token).

   ```python
   for chunk in model.stream("Explain quantum computing in 2 sentences"):
       print(chunk.content, end="", flush=True)
   ```

3. **Batch Processing (`model.batch()`)**:
   - *Concept*: Dispatches multiple independent prompts in parallel using async thread pools.
   - *Performance Benefit*: Drastically reduces total latency and increases request throughput compared to sequential `for` loops.

   ```python
   responses = model.batch(["What is 2+2?", "What is 10*5?", "What is 100/4?"])
   ```

---

### 3. `3-tools.ipynb` – Tool Anatomy, Schemas & Both Tool Binding Methods
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/82017e3a-16fe-48eb-8943-f38efc611922" />

#### 🧠 Theory & Core Concepts
A **Tool** is a pairing of:
1. **JSON Schema**: Contains the function name, docstring description, and argument parameter types.
2. **Execution Logic**: The underlying Python function or coroutine executed when invoked.

The `@tool` decorator automatically inspects Python type hints (`city: str`) and Google/Sphinx docstrings to auto-generate the JSON schema expected by LLM tool-calling APIs.

#### 🛠️ Both Methods to Add & Bind Tools in LangChain

| Feature | Method 1: `model.bind_tools()` | Method 2: `create_agent(tools=[...])` |
| :--- | :--- | :--- |
| **Execution Loop** | Manual (Developer invokes tool function) | Automatic (LangGraph engine invokes tool function) |
| **Message State** | Manual `ToolMessage` creation & append | Automatic `ToolMessage` state tracking |
| **Control Level** | Fine-grained (Custom UI callbacks, single-step) | High-level (Multi-step autonomous agent execution) |

##### Method 1: Direct Model Binding (`model.bind_tools()`)
```python
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

model = init_chat_model("gpt-4o-mini")
model_with_tools = model.bind_tools([get_weather])

# Step 1: Model generates tool calls
messages = [{"role": "user", "content": "What's the weather in Boston?"}]
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)

# Step 2: Execute tools and collect results
for tool_call in ai_msg.tool_calls:
    # Execute the tool with the generated arguments
    tool_result = get_weather.invoke(tool_call)
    messages.append(tool_result)

# Step 3: Pass results back to model for final response
final_response = model_with_tools.invoke(messages)
print(final_response.content)
# "The weather in Boston is sunny."
```

##### Method 2: Automatic Agent Execution Loop (`create_agent(tools=[...])`)
```python
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

# Agent automatically executes the loop: LLM -> Tool Call -> Function Exec -> ToolMessage -> Final Answer
agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_weather],
    system_prompt="You are a helpful assistant."
)
result = agent.invoke({"messages": [{"role": "user", "content": "What's the weather in Boston?"}]})
```

---

### 4. `4-messages.ipynb` – Canonical Message State & Token Usage Metadata

#### 🧠 Theory & Core Concepts
Messages are the fundamental unit of context in LangChain. They represent multi-turn conversation state and carry content, roles, and provider metadata across APIs.

- **Text Prompts vs. Message Prompts**:
  - *Text Prompts*: Standalone strings for simple, single-turn tasks.
  - *Message Prompts*: Structured list of `BaseMessage` objects required for multi-turn chat memory and agent tool-calling loops.

#### 💬 The 4 Canonical Message Types

| Message Class | Role | Purpose & Contents |
| :--- | :--- | :--- |
| **`SystemMessage`** | `system` | Instructions setting persona, tone, rules, and guardrails. |
| **`HumanMessage`** | `user` | User inputs (supports multimodal text, images, audio, files). |
| **`AIMessage`** | `assistant` | Model output (text, reasoning tokens, and `tool_calls` payload). |
| **`ToolMessage`** | `tool` | Output returned by a tool execution, mapped via `tool_call_id`. |

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

messages = [
    SystemMessage("You are a helpful financial assistant."),
    HumanMessage("What is the stock price of Apple?"),
    AIMessage(content="", tool_calls=[{"name": "get_stock_price", "args": {"ticker": "AAPL"}, "id": "call_999"}]),
    ToolMessage(content="$225.50", tool_call_id="call_999")
]

response = model.invoke(messages)
print(response.usage_metadata)  # Contains token usage details
```

---

### 5. `5-structuredoutput.ipynb` – Enforced Schema Parsing & Validation

#### 🧠 Theory & Core Concepts
**Structured Output** guarantees that an LLM responds matching a strict schema (JSON/Pydantic), eliminating parsing failures in downstream production code.

#### 📐 Supported Schema Enforcers

1. **Pydantic (`BaseModel`)**: Full runtime field validation, default values, and rich field descriptions (`Field(description=...)`).
2. **TypedDict (`TypedDict`)**: Lightweight Python built-in typing using `Annotated[T, Field(description=...)]`.
3. **Dataclass (`@dataclass`)**: Standard Python data container.

```python
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated

# Option A: Pydantic Schema
class Movie(BaseModel):
    title: str = Field(description="Title of the movie")
    year: int = Field(description="Release year")

structured_model = model.with_structured_output(Movie)
movie_obj = structured_model.invoke("Provide details about Inception")
# Returns: Movie(title='Inception', year=2010)

# Option B: TypedDict Schema
class MovieDict(TypedDict):
    title: Annotated[str, Field(description="Title of the movie")]
    year: Annotated[int, Field(description="Release year")]

structured_dict_model = model.with_structured_output(MovieDict)
```

- **`include_raw=True`**: Returns a dictionary with `{"raw": AIMessage, "parsed": SchemaObject, "parsing_error": None}` for debugging and tracing raw model tokens.

---

### 6. `6-middleware.ipynb` – Stateful Middleware, Memory Compression & Human-in-the-Loop

#### 🧠 Theory & Core Concepts
**Middleware** intercept and modify internal agent execution steps. They are essential for:
- Logging, analytics, and token cost control.
- Guardrails, PII masking, and output safety filtering.
- Automated memory compression and human approval checkpoints.

#### 1. Summarization Middleware (`SummarizationMiddleware`)
- *Problem*: Long-running multi-turn agent conversations exceed context window limits and consume excessive tokens.
- *Solution*: Automatically compresses older conversation turns into a summarized context block once a message count or token limit is reached, while keeping recent messages intact.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-4o-mini",
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("messages", 10), # Summarize after 10 messages
            keep=("messages", 4)       # Keep latest 4 messages
        )
    ]
)
```

#### 2. Human-In-The-Loop Middleware (`HumanInTheLoopMiddleware`)
- *Problem*: High-stakes tool executions (e.g. database deletes, financial transactions, sending emails) require human oversight before execution.
- *Solution*: Pauses agent execution before executing specified tools. The agent state is persisted using a checkpointer (`InMemorySaver`), and waits for a human approval/rejection/edit signal (`Command`).

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-4o-mini",
    tools=[send_email_tool, read_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_before=["send_email_tool"]  # Pause before sending email
        )
    ]
)
```
