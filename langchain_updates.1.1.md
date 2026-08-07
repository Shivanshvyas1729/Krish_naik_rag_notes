# LangChain v1.1 & LangGraph Agent Architecture (Exhaustive Master Reference)

---

## 📋 Overview & Module Index

This document is the **definitive, production-grade technical manual** for **LangChain v1.1 / 1.x** built on top of the **LangGraph** execution engine. It covers every concept, API signature, parameter, theory note, and code cell across all 6 notebooks in `08_langchain_updated_version1.1`.

### 📦 Workspace Package Versioning & Import Conventions
```python
import langchain
print("LangChain Version:", langchain.__version__)  # e.g., 0.3.x / 1.1.x
```

- **LangChain Message Imports**: Available via `from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage` or `from langchain_core.messages import ...`.
- **Model Initializer**: `from langchain.chat_models import init_chat_model`.
- **Agent Creator**: `from langchain.agents import create_agent`.
- **Middleware**: `from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware`.
- **Checkpointers**: `from langgraph.checkpoint.memory import InMemorySaver`.
- **Command Resumption**: `from langgraph.types import Command`.

---

## 📚 Master Index of Notebooks & Key Technical APIs

| Notebook | Topic & Core Focus | Key APIs & Special Parameters Covered |
| :--- | :--- | :--- |
| **`1-langchainintro.ipynb`** | Agent Foundations & Stateful Graph Engine | `create_agent()`, `@tool`, `agent.invoke()`, string & list prompts, `response["messages"]` |
| **`2-modelintegration.ipynb`** | Universal Model Initializer, Multi-Provider, Streaming & Batching | `init_chat_model()`, `ChatOpenAI()`, `ChatGroq()`, `ChatGoogleGenerativeAI()`, `model.stream()`, `model.batch()`, `max_concurrency` |
| **`3-tools.ipynb`** | Tool Definition, Schemas, Manual Loop & Agent Loop | `@tool`, `model.bind_tools()`, `ai_msg.tool_calls`, `get_weather.invoke()`, 3-Step Manual Loop |
| **`4-messages.ipynb`** | Canonical Message Schema, Text vs. Message Prompts & Token Metadata | `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`, `tool_call_id`, `usage_metadata`, `response.text` |
| **`5-structuredoutput.ipynb`** | Enforced Schema Parsing, `include_raw`, Nested Schemas, `model.profile` & Agent `response_format` | `with_structured_output(..., include_raw=True)`, `response_format`, `Pydantic`, `TypedDict`, `Annotated`, `@dataclass`, `model.profile`, `result["structured_response"]` |
| **`6-middleware.ipynb`** | Agent Middleware, Token Optimization, Summarization & Human-in-the-Loop | `SummarizationMiddleware`, `HumanInTheLoopMiddleware`, `InMemorySaver`, `Command`, `trigger=("messages"|"fraction")`, `interrupt_on={tool: False}`, `allowed_decisions=["approve","edit","reject"]` |

---

## 📖 Complete Cell-by-Cell Theory & Deep Dives

---

### 1. `1-langchainintro.ipynb` – Agent Foundations & High-Level Architecture

#### 🧠 Theory & Core Principles
An **AI Agent** utilizes a Large Language Model (LLM) as a dynamic reasoning engine to process inputs, select external tools, extract parameters, execute operations, and synthesize answers.

- **Legacy (`AgentExecutor`) vs. LangChain v1.1 (`create_agent`)**:
  - *Legacy (`AgentExecutor`)*: Imperial Python loop architecture requiring manual memory array passing and hardcoded execution limits.
  - *LangChain v1.1 (`create_agent`)*: Constructs a stateful, compiled execution graph powered by **LangGraph** under the hood. It natively supports cyclic agent reasoning loops, state checkpointers, automatic message tracking, and tool failure recovery.

#### 💻 Code Implementation & Input Patterns

```python
import os
from dotenv import load_dotenv
import langchain
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

print("LangChain Version:", langchain.__version__)

# Define tool with docstring description & type hints
@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

# Create stateful compiled agent graph
agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_weather],
    system_prompt="You are a helpful assistant."
)

# Pattern A: Invoke with Message List Dictionary
response_dict = agent.invoke({
    "messages": [{"role": "user", "content": "What is the weather like in New York?"}]
})
print("Messages in Graph:", response_dict["messages"])

# Pattern B: Invoke with Simple Input String
response_str = agent.invoke({
    "messages": "What is the weather in New York"
})
```

#### 🔍 Output Execution Graph Structure
The returned dictionary contains `"messages"` representing the complete execution trajectory:
1. `HumanMessage`: `"What is the weather like in New York?"`
2. `AIMessage`: `content=""`, `tool_calls=[{"name": "get_weather", "args": {"city": "New York"}, "id": "call_123"}]`
3. `ToolMessage`: `content="The weather in New York is sunny."`, `tool_call_id="call_123"`
4. `AIMessage`: `content="The weather in New York is currently sunny."`

---

### 2. `2-modelintegration.ipynb` – Universal Model Provider Loading, Streaming & Batching

#### 🧠 Theory & Core Principles
LangChain v1.1 decouples LLM vendor bindings from application code through a universal factory initializer and standardized invocation contracts.

1. **Universal Model Initializer (`init_chat_model`)**:
   Instead of vendor-locked imports (`ChatOpenAI`, `ChatGroq`, `ChatGoogleGenerativeAI`), `init_chat_model()` instantiates any chat LLM using string specifiers (`"gpt-4.1"`, `"google_genai:gemini-2.5-flash"`, `"groq:qwen/qwen3-32b"`).
2. **Streaming Output (`model.stream()`)**:
   LLMs generate text token by token. `model.stream()` uses HTTP chunked transfer encoding to yield `AIMessageChunk` objects in real time, eliminating user latency waiting time (**TTFT**).
3. **Batch Execution (`model.batch()`) & Concurrency Control (`max_concurrency`)**:
   `model.batch()` dispatches multiple independent queries in parallel using async/thread worker pools. Passing `config={'max_concurrency': N}` limits active parallel HTTP connections to prevent API rate limit failures.

#### 💻 Code Implementation

```python
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ==========================================
# 1. Universal vs. Direct Provider Loading
# ==========================================

# OpenAI (Universal vs. Direct)
model_openai = init_chat_model("gpt-4.1")
model_openai_direct = ChatOpenAI(model="gpt-4.1")

# Google Gemini (Universal vs. Direct)
model_gemini = init_chat_model("google_genai:gemini-2.5-flash")
model_gemini_direct = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

# GROQ (Universal vs. Direct)
model_groq = init_chat_model("groq:qwen/qwen3-32b")
model_groq_direct = ChatGroq(model="qwen/qwen3-32b")

# ==========================================
# 2. Streaming Real-Time Output Chunking
# ==========================================
print("--- Streaming Output ---")
for chunk in model_groq.stream("Write me a 200 words paragraph on Artificial Intelligence"):
    # Accessing .text or .content property on AIMessageChunk
    print(chunk.text, end="|", flush=True)

# ==========================================
# 3. Batching with max_concurrency Cap
# ==========================================
print("\n--- Parallel Batch Execution ---")
questions = [
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
]

# Batch execution with concurrency control
responses = model_groq.batch(
    questions,
    config={"max_concurrency": 5}  # Limits parallel calls to 5
)

for resp in responses:
    print("Response Length:", len(resp.content))
```

---

### 3. `3-tools.ipynb` – Tool Anatomy, Schemas & Both Tool Binding Methods

#### 🧠 Theory & Core Principles
A **Tool** is a pairing of:
1. **JSON Schema**: Function name, docstring description, and argument definitions.
2. **Execution Logic**: The underlying Python function or coroutine executed when triggered.

The `@tool` decorator inspects function signatures (`location: str`) and docstrings (`"""Get the weather at a location"""`) to automatically generate OpenAI-compatible JSON schemas.

#### 🛠️ Comparison: Method 1 vs. Method 2 Tool Execution

| Feature | Method 1: `model.bind_tools()` | Method 2: `create_agent(tools=[...])` |
| :--- | :--- | :--- |
| **Execution Loop** | **Manual** (Developer loops over `ai_msg.tool_calls`) | **Automatic** (LangGraph handles loop & recursion) |
| **Message State** | **Manual** (`ToolMessage` creation & array `.append()`) | **Automatic** (Graph state updates automatically) |
| **Control Level** | Fine-grained (Human verification, custom UI handlers) | High-level (Autonomous multi-step agent execution) |
| **Tool Execution** | Explicit `tool_func.invoke(tool_call)` call | Framework handles tool invocation internally |

#### 💻 Code Implementation

```python
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import ToolMessage

# Define Tool via @tool Decorator
@tool
def get_weather(location: str) -> str:
    """Get the weather at a location"""
    return f"It's sunny in {location}"

model = init_chat_model("groq:qwen/qwen3-32b")

# =========================================================
# Method 1: Direct Model Binding & 3-Step Manual Loop
# =========================================================
model_with_tools = model.bind_tools([get_weather])

# Step 1: Model generates tool calls
messages = [{"role": "user", "content": "What's the weather in Boston?"}]
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)

# Inspect model generated tool_calls structure
for tool_call in ai_msg.tool_calls:
    print(f"Tool Name: {tool_call['name']}")
    print(f"Tool Args: {tool_call['args']}")
    print(f"Tool ID:   {tool_call['id']}")

# Step 2: Execute tool functions manually and collect results
for tool_call in ai_msg.tool_calls:
    tool_result = get_weather.invoke(tool_call)  # Returns result string or ToolMessage
    if not isinstance(tool_result, ToolMessage):
        tool_result = ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
    messages.append(tool_result)

# Step 3: Pass updated message history back to model for final answer synthesis
final_response = model_with_tools.invoke(messages)
print("Final Response Text:", final_response.text)
# "The current weather in Boston is 72F and sunny."

# =========================================================
# Method 2: Automatic Agent Execution Loop
# =========================================================
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_weather],
    system_prompt="You are a helpful assistant."
)

agent_res = agent.invoke({"messages": "What's the weather in Boston?"})
print("Automated Agent Result:", agent_res["messages"][-1].content)
```

---

### 4. `4-messages.ipynb` – Canonical Message State, Prompts & Token Usage Metadata

#### 🧠 Theory & Core Principles
Messages carry content, role types, and provider metadata across conversational turns.

- **Text Prompts vs. Message Prompts**:
  - *Text Prompts*: Standalone strings for simple, single-turn requests with zero history retention.
  - *Message Prompts*: Structured arrays of `BaseMessage` objects required for multi-turn history and agent loops.

#### 💬 The 4 Canonical Message Types

| Message Class | Role | Purpose & Contents |
| :--- | :--- | :--- |
| **`SystemMessage`** | `system` | Sets persona, instructions, system rules, and safety guardrails. |
| **`HumanMessage`** | `user` | User inputs (supports text & multimodal content). |
| **`AIMessage`** | `assistant` | Model generated response containing text content, `tool_calls` payload, and metadata. |
| **`ToolMessage`** | `tool` | Output returned by a tool call execution. **Must include `tool_call_id` matching `tool_call['id']`**. |

#### 💻 Code Implementation

```python
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

model = init_chat_model("groq:qwen/qwen3-32b")

# 1. SystemMessage, HumanMessage & Manual AIMessage Context Injection
system_msg = SystemMessage("You are a helpful travel assistant. Keep responses brief.")
human_msg = HumanMessage("What should I pack for a 3-day trip to Paris in spring?")
ai_injected_msg = AIMessage("I'd be happy to help you with that question!")

messages = [
    system_msg,
    human_msg,
    ai_injected_msg,  # Inserted into conversation history as if generated by model
    HumanMessage("Great! What is 2+2?")
]

response = model.invoke(messages)
print("Response Content:", response.content)

# 2. Token Usage Metadata Inspection
print("Usage Metadata:", response.usage_metadata)
# Output: {'input_tokens': 45, 'output_tokens': 12, 'total_tokens': 57}

# 3. Tool Call & ToolMessage Mapping Pattern
ai_message = AIMessage(
    content=[],
    tool_calls=[{
        "name": "get_weather",
        "args": {"location": "San Francisco"},
        "id": "call_123"
    }]
)

tool_message = ToolMessage(
    content="Sunny, 72F",
    tool_call_id="call_123"  # Must match the call ID
)

chat_trajectory = [
    HumanMessage("What's the weather in San Francisco?"),
    ai_message,
    tool_message
]

final_tool_res = model.invoke(chat_trajectory)
print("Final Grounded Response:", final_tool_res.content)
```

---

### 5. `5-structuredoutput.ipynb` – Enforced Schema Parsing, `include_raw`, Nested Schemas, `model.profile` & Agent `response_format`

#### 💡 Deep Dive: TypedDict vs. Pydantic, Runtime Validation & LLM Retry Mechanics

##### 1. Why Use `TypedDict` When We Have `Pydantic`?
While `Pydantic` is feature-packed with runtime data validation, `TypedDict` serves distinct, complementary engineering purposes:
- **No Runtime Overhead**: `TypedDict` does not validate data types when instantiated during program execution. It incurs zero CPU overhead, making it ideal when parsing performance is critical and input sources are already trusted.
- **Lightweight Standard Library Dependency**: `TypedDict` is built into standard Python (`typing`/`typing_extensions`), avoiding extra external dependencies or package footprint.
- **Static Analysis & IDE Introspectability**: Provides type hints for static checkers like `mypy` and IDE autocompletion without raising execution errors if minor payload mismatches occur.
- **Choice Rule**: Use **`Pydantic`** when ingesting untrusted data (user input, external API webhooks, LLM outputs) where strict validation is mandatory. Use **`TypedDict`** for simple internal state definitions or lightweight dictionary interfaces.

##### 2. What is Runtime Validation?
**Runtime Validation** is the process of inspecting and validating data types and constraints *at the exact moment code executes*, rather than relying on pre-execution static checks.
- **`TypedDict` Behavior**: Does **not** perform runtime validation. Constructing `MovieDict(year="2024")` (passing a string where an `int` was defined) executes silently without throwing a `TypeError`.
- **`Pydantic` Behavior**: Performs **strict runtime validation**. Instantiating `Movie(year="invalid_str")` immediately halts execution and raises a `pydantic.ValidationError` containing detailed line-by-line error reports.

##### 3. Does Pydantic Automatically Retry on Validation Error with LLMs?
**No.** When Pydantic raises a `ValidationError`, the LLM does **NOT** automatically retry by default. The exception is raised directly in your application thread.

**Developer Responsibilities & LLM Retry Patterns**:
1. **Manual Exception Handling**: You can wrap `model.invoke()` in a `try...except ValidationError as e:` block.
2. **Automated Error Feedback & Re-Prompting**: To achieve automatic retries, you catch the `ValidationError`, extract `e.json()`, and feed the error string back into the model prompt so the LLM self-corrects:
   ```python
   from pydantic import ValidationError
   from langchain_core.messages import HumanMessage, AIMessage

   messages = [HumanMessage(content="Extract details: Inception (unknown_year)")]

   for attempt in range(3):
       try:
           structured_output = model_with_structure.invoke(messages)
           break  # Validation passed!
       except ValidationError as e:
           # Re-prompt model with explicit validation error feedback
           messages.append(AIMessage(content="[Invalid Output]"))
           messages.append(HumanMessage(content=f"Your previous output failed validation: {e}. Please fix type errors and retry."))
   ```
3. **LangChain Retry Utilities**: LangChain provides built-in parsers like `OutputFixingParser` and `RetryWithErrorOutputParser` that automate this error feedback retry loop.

#### 📐 Supported Schema Enforcers & Special Parameters

1. **Pydantic (`BaseModel`)**: Full runtime validation, default values, nested models, and field descriptions (`Field(description=...)`).
2. **`include_raw=True` Parameter (CRITICAL KEY FEATURE)**:
   - Calling `with_structured_output(Movie, include_raw=True)` returns a dictionary with 3 top-level keys:
     - `'raw'`: The full raw `AIMessage` returned by the LLM (useful for inspecting token usage or raw text).
     - `'parsed'`: The instantiated & validated schema object.
     - `'parsing_error'`: `None` if successful, or the parsing exception if validation failed.
3. **Nested Pydantic & TypedDict Schemas**: Sub-models (`Actor`) nested inside parent schemas (`MovieDetails`) via `list[Actor]` or optional fields (`float | None = Field(...)`).
4. **`model.profile` Attribute**: Returns model provider capability profiling info.
5. **Agent Integration (`response_format`)**: Passing a schema class to `create_agent(response_format=Schema)` auto-selects native provider strategies and places the validated object inside `result["structured_response"]`.

#### 💻 Code Implementation

```python
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated
from dataclasses import dataclass
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

model = init_chat_model("groq:qwen/qwen3-32b")

# Inspect Model Profile
print("Model Profile Attributes:", getattr(model, "profile", "N/A"))

# =========================================================
# 1. Pydantic Schema with include_raw=True
# =========================================================
class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(description="The title of the movie")
    year: int = Field(description="The year the movie was released")
    director: str = Field(description="The director of the movie")
    rating: float = Field(description="The movie's rating out of 10")

# Enabling include_raw=True returns {'raw': AIMessage, 'parsed': Movie, 'parsing_error': None}
model_with_raw = model.with_structured_output(Movie, include_raw=True)
response_raw = model_with_raw.invoke("Provide details about the movie Inception")

print("Raw AIMessage:", response_raw["raw"])
print("Parsed Movie Object:", response_raw["parsed"])
print("Parsing Error:", response_raw["parsing_error"])

# =========================================================
# 2. Nested Pydantic & Nested TypedDict Schemas
# =========================================================

# --- A. Nested Pydantic Models ---
class ActorPydantic(BaseModel):
    name: str = Field(description="Name of the actor")
    role: str = Field(description="Character role played by the actor")

class MovieDetailsPydantic(BaseModel):
    title: str
    year: int
    cast: list[ActorPydantic]  # Nested list of Actor Pydantic objects
    genres: list[str]
    budget: float | None = Field(None, description="Budget in millions USD")

model_nested_pyd = model.with_structured_output(MovieDetailsPydantic)
res_pyd = model_nested_pyd.invoke("Provide details about the movie Inception")
print("Lead Actor Pydantic:", res_pyd.cast[0].name)

# --- B. Nested TypedDict Models ---
class ActorTypedDict(TypedDict):
    name: str
    role: str

class MovieDetailsTypedDict(TypedDict):
    title: str
    year: int
    cast: list[ActorTypedDict]  # Nested list of Actor TypedDict dictionaries
    genres: list[str]
    budget: float | None = Field(None, description="Budget in millions USD")

model_nested_td = model.with_structured_output(MovieDetailsTypedDict)
res_td = model_nested_td.invoke("Provide details about the movie Avengers")
print("Lead Actor TypedDict:", res_td["cast"][0]["name"])

# =========================================================
# 3. TypedDict with Annotated Metadata
# =========================================================
class MovieDict(TypedDict):
    """A movie with details."""
    title: Annotated[str, Field(description="The title of the movie")]
    year: Annotated[int, Field(description="The year the movie was released")]
    director: Annotated[str, Field(description="The director of the movie")]
    rating: Annotated[float, Field(description="The movie's rating out of 10")]

model_typeddict = model.with_structured_output(MovieDict)
dict_response = model_typeddict.invoke("Provide details about Avengers")

# =========================================================
# 4. Agent Integration via response_format
# =========================================================

# Option A: Pydantic Schema
class ContactInfoPydantic(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")

agent_pyd = create_agent(model="gpt-4o-mini", response_format=ContactInfoPydantic)
res_agent_pyd = agent_pyd.invoke({"messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]})
print("Structured Response Pydantic:", res_agent_pyd["structured_response"])

# Option B: TypedDict Schema
class ContactInfoTypedDict(TypedDict):
    """Contact information for a person."""
    name: str
    email: str
    phone: str

agent_td = create_agent(model="gpt-4o-mini", response_format=ContactInfoTypedDict)
res_agent_td = agent_td.invoke({"messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]})
print("Structured Response TypedDict:", res_agent_td["structured_response"])

# Option C: DataClass Schema
@dataclass
class ContactInfoDataClass:
    """Contact information for a person."""
    name: str
    email: str
    phone: str

agent_dc = create_agent(model="gpt-4o-mini", response_format=ContactInfoDataClass)
res_agent_dc = agent_dc.invoke({"messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]})
print("Structured Response DataClass:", res_agent_dc["structured_response"])
```

---

### 6. `6-middleware.ipynb` – Stateful Middleware, Token Optimization & Human-in-the-Loop Operations

#### 🧠 Theory & Core Principles
**Middleware** intercept, modify, or pause internal agent execution steps. Useful for:
- Logging, analytics, and token expenditure monitoring.
- Guardrails, PII masking, and output safety policies.
- Automated conversation history compression.
- Human approval checkpoints before executing sensitive tools.

#### 1. Summarization Middleware (`SummarizationMiddleware`)
- **Problem**: Multi-turn agent chats accumulate token history, eventually exceeding model context windows.
- **Solution**: Triggers a background summarization pass that compresses older turns into a single summary `SystemMessage` block.
- **Trigger Strategies**:
  1. `trigger=("messages", 10)`, `keep=("messages", 4)`: Triggers when conversation reaches 10 messages; keeps the 4 most recent messages intact.
  2. `trigger=("fraction", 0.005)`, `keep=("fraction", 0.002)`: Triggers when message tokens reach 0.5% of total model context (e.g. ~640 tokens for 128k context), compressing down to 0.2% (~256 tokens).
- **Token Counter Heuristic**: `count_tokens(messages) = sum(len(str(m.content)) for m in messages) // 4` (4 chars ≈ 1 token).

#### 2. Human-In-The-Loop Middleware (`HumanInTheLoopMiddleware`)
- **Problem**: Operations like sending emails, financial transfers, or database writes require human oversight before executing.
- **Solution**: Pauses agent execution before triggering targeted tools. Saves graph state via a checkpointer (`InMemorySaver`), waiting for a human resume signal via `Command(resume=...)`.
- **`interrupt_on` Configuration**:
  ```python
  interrupt_on={
      "send_email_tool": {"allowed_decisions": ["approve", "edit", "reject"]},
      "read_email_tool": False  # Disable interrupt for read-only operations
  }
  ```
- **Allowed Human Decisions**:
  - `"approve"`: Proceeds with tool execution unchanged.
  - `"reject"`: Cancels tool execution and returns cancellation message to the model.
  - `"edit"`: Overrides tool arguments before execution (`edited_action={"name": "send_email_tool", "args": {...}}`).

#### 💻 Code Implementation

```python
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain.messages import HumanMessage
from langchain.tools import tool

load_dotenv()

# Token Counter Helper
def count_tokens(messages):
    return sum(len(str(m.content)) for m in messages) // 4  # 4 chars approx 1 token

# =========================================================
# 1. Summarization Middleware (Message & Fraction Triggers)
# =========================================================

# Message-Count Trigger Agent
agent_sum_msg = create_agent(
    model="gpt-4o-mini",
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("messages", 10),
            keep=("messages", 4)
        )
    ]
)

# Token Fraction Trigger Agent
agent_sum_frac = create_agent(
    model="gpt-4o-mini",
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("fraction", 0.005),  # 0.5% of 128k context (~640 tokens)
            keep=("fraction", 0.002)      # 0.2% of 128k context (~256 tokens)
        )
    ]
)

thread_config = {"configurable": {"thread_id": "test-session-1"}}

# Run multiple turns to trigger summarization
questions = ["What is 2+2?", "What is 10*5?", "What is 100/4?", "What is 15-7?", "What is 3*3?", "What is 4*4?"]
for q in questions:
    res = agent_sum_msg.invoke({"messages": [HumanMessage(content=q)]}, config=thread_config)

# =========================================================
# 2. Human-in-the-Loop Middleware (Approve, Reject, Edit)
# =========================================================

@tool
def read_email_tool(email_id: str) -> str:
    """Mock function to read an email by its ID."""
    return f"Email content for ID: {email_id}"

@tool
def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Mock function to send an email."""
    return f"Email sent to {recipient} with subject '{subject}'"

agent_hitl = create_agent(
    model="gpt-4o",
    tools=[read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"]
                },
                "read_email_tool": False  # Read ops do not require approval
            }
        )
    ]
)

# ---------------------------------------------------------
# Decision 1: APPROVE
# ---------------------------------------------------------
config_approve = {"configurable": {"thread_id": "test-approve"}}
res_app = agent_hitl.invoke(
    {"messages": [HumanMessage(content="Send email to john@test.com with subject 'Hello' and body 'How are you?'")]},
    config=config_approve
)

if "__interrupt__" in res_app:
    print(" Execution Paused! Resuming with APPROVE...")
    res_app = agent_hitl.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config_approve
    )
    print("Approve Final Result:", res_app["messages"][-1].content)

# ---------------------------------------------------------
# Decision 2: REJECT
# ---------------------------------------------------------
config_reject = {"configurable": {"thread_id": "test-reject"}}
res_rej = agent_hitl.invoke(
    {"messages": [HumanMessage(content="Send email to john@test.com with subject 'Hello' and body 'How are you?'")]},
    config=config_reject
)

if "__interrupt__" in res_rej:
    print(" Execution Paused! Resuming with REJECT...")
    res_rej = agent_hitl.invoke(
        Command(resume={"decisions": [{"type": "reject"}]}),
        config=config_reject
    )
    print("Reject Final Result:", res_rej["messages"][-1].content)

# ---------------------------------------------------------
# Decision 3: EDIT
# ---------------------------------------------------------
config_edit = {"configurable": {"thread_id": "test-edit"}}
res_edit = agent_hitl.invoke(
    {"messages": [HumanMessage(content="Send email to wrong@email.com with subject 'Test' and body 'Hello'")]},
    config=config_edit
)

if "__interrupt__" in res_edit:
    print(" Execution Paused! Resuming with EDIT...")
    res_edit = agent_hitl.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": "edit",
                        "edited_action": {
                            "name": "send_email_tool",
                            "args": {
                                "recipient": "correct@email.com",
                                "subject": "Corrected Subject",
                                "body": "This was edited by human before sending"
                            }
                        }
                    }
                ]
            }
        ),
        config=config_edit
    )
    print("Edit Final Result:", res_edit["messages"][-1].content)
```

---

## 🎨 Appendix: Why the CLIP Model & Processor are Essential in Multimodal RAG

In a Multimodal Retrieval-Augmented Generation (RAG) system (such as processing complex enterprise documents containing both text passages and visual charts/diagrams), the **CLIP Model** (`CLIPModel`) and **CLIP Processor** (`CLIPProcessor`) are foundational components for the following core reasons:

### 1. Joint Vector Embedding Space
CLIP converts both visual images (pixels) and text strings into vectors within the **exact same 512-dimensional vector space**. This shared embedding space enables cross-modal similarity search: a natural language text query (e.g., *"Show Q1 revenue growth chart"*) directly matches image embedding vectors in a vector database (`FAISS`) even if no text caption was extracted from that image.

### 2. Unified Multimodal Architecture
Rather than maintaining separate NLP text embedding models and Computer Vision feature extractors, CLIP processes both text and image modalities using a single unified model architecture, simplifying the system design and eliminating vector dimension alignment issues.

### 3. Data Preprocessing (`CLIPProcessor`)
The `CLIPProcessor` prepares raw input data before feeding it into the CLIP model:
- **Image Preprocessing**: Resizes, crops, and normalizes raw image pixel arrays (converting CMYK/RGBA bytes to standardized RGB floating-point tensors).
- **Text Preprocessing**: Tokenizes and truncates input queries into 77-token sequences compatible with CLIP's text transformer encoder.

### 4. Cross-Modal Contextual Alignment
CLIP is contrastively pre-trained on hundreds of millions of image-caption pairs. It possesses deep semantic understanding of the relationship between textual concepts and visual elements, enabling rich cross-modal retrieval for Vision LLMs (`GPT-4o`).

