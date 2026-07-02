# RAG (Retrieval-Augmented Generation) Short Notes (1-9.1)

---

## 1. Data Ingestion & Splitting (`1-dataingestion.ipynb`)
### Imports
```python
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
```
### How to Use
```python
# Load single text file and split recursively
loader = TextLoader("data/sample.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
```
### What They Do
*   `Document`: LangChain's base class storing text (`page_content`) and metadata (`metadata` dict).
*   `RecursiveCharacterTextSplitter`: Splits text recursively using a hierarchy of separators (e.g., `\n\n`, `\n`, `" "`). **Best default splitter** as it preserves paragraphs/sentences.
*   `CharacterTextSplitter`: Splits text rigidly based on a single character separator.
*   `TokenTextSplitter`: Splits text strictly by token count (using `tiktoken`) to fit LLM context limits.
*   `TextLoader` / `DirectoryLoader`: Loads a single plain text file / bulk-loads matching files from a folder.
*   **Key Concept**: Chunk overlap is crucial to maintain contextual information across split boundaries.

---

## 2. PDF Parsing (`2-dataparsingpdf.ipynb`)
### Imports
```python
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
```
### How to Use
```python
# Load PDF page-by-page (PyMuPDF is fast and layout-accurate)
loader = PyMuPDFLoader("data/sample.pdf")
pages = loader.load()
```
### What They Do
*   `PyPDFLoader`: Simple, pure Python loader that extracts text page-by-page.
*   `PyMuPDFLoader`: C-based, high-accuracy PDF text extractor. **Extremely fast** and better at handling complex multi-column layouts and images.
*   **Key Concept**: PyMuPDF is much faster. Always implement a text cleaning step to fix ligatures (`ﬁ`, `ﬂ`) and extra spaces injected during parsing.

---

## 3. Word Document Parsing (`3-dataparsingdoc.ipynb`)
### Imports
```python
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredWordDocumentLoader
from unstructured.partition.docx import partition_docx
```
### How to Use
```python
# Simple text extraction
loader = Docx2txtLoader("data/sample.docx")
docs = loader.load()

# Element-based structured extraction
unstructured_loader = UnstructuredWordDocumentLoader("data/sample.docx", mode="elements", strategy="fast")
element_docs = unstructured_loader.load()
```
### What They Do
*   `Docx2txtLoader`: Fast, lightweight loader that extracts plain text from `.docx` files.
*   `UnstructuredWordDocumentLoader`: Uses the `unstructured` library to partition documents into elements (Titles, NarrativeText, Tables). Using `mode="elements"` yields separate documents per logical section.
*   **Key Concept**: Use `Docx2txtLoader` for simple files, and `Unstructured` (with `strategy="fast"`) if you need to filter out headers/footers/page numbers by category metadata.

---

## 4. CSV & Excel Structured Parsing (`4-csvexcelparsing.ipynb`)
### Imports
```python
from langchain_community.document_loaders import CSVLoader, UnstructuredExcelLoader
import pandas as pd
```
### How to Use
```python
# Load CSV (each row is loaded as a separate Document object)
loader = CSVLoader("data/products.csv", source_column="Product")
docs = loader.load()
```
### What They Do
*   `CSVLoader`: Creates a Document object for each row of a CSV, writing columns as key-value text lines.
*   `UnstructuredExcelLoader`: Loads sheets and tables as text elements.
*   `pandas.DataFrame`: Used for custom CSV/Excel processing before converting to LangChain Documents.
*   **Key Concept**: Avoid embedding huge database-like tables row-by-row in a vector store. Convert rows to structured natural language summaries (e.g., `"Product X is category Y priced at Z"`) for better retrieval.

---

## 5. JSON Parsing (`5-jsonparsing.ipynb`)
### Imports
```python
from langchain_community.document_loaders import JSONLoader
```
### How to Use
```python
# Load specific values from nested JSON using jq path queries
loader = JSONLoader("data/company.json", jq_schema=".employees[].role", text_content=True)
docs = loader.load()
```
### What They Do
*   `JSONLoader`: Loads JSON/JSONL files. Uses a `jq_schema` path filter to extract specific nested attributes/fields as separate Documents.
*   **Key Concept**: Avoid embedding raw JSON syntax symbols (`{}[]"`). Construct clean natural language summaries for indexing, and keep raw JSON fields in metadata for precise filtering.

---

## 6. Database Parsing (`6-databaseparsing.ipynb`)
### Imports
```python
from langchain_community.utilities import SQLDatabase
from langchain_community.document_loaders import SQLDatabaseLoader
```
### How to Use
```python
# Connect to DB and load specific query outputs as Documents
db = SQLDatabase.from_uri("sqlite:///data/company.db")
loader = SQLDatabaseLoader(query="SELECT name, role FROM employees", db=db)
docs = loader.load()
```
### What They Do
*   `SQLDatabase`: Connects to relational databases (SQLite, PostgreSQL, etc.) and provides DDL schema metadata.
*   `SQLDatabaseLoader`: Runs queries on a database and loads each row as a LangChain Document.
*   **Key Concept**: Embedding an entire relational database is an anti-pattern. Instead, use a Text-to-SQL agent workflow (passing DDL schemas to LLM) or connect using SQL connections with strict **Read-Only** permissions.

---

## 7. Embedding Models (`7.0-embedding.ipynb` & `7.1-openaiembeddings.ipynb`)
### Imports
```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
```
### How to Use
```python
# Local HuggingFace Embeddings
hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_query = hf_embeddings.embed_query("your query text")

# API-Based OpenAI Embeddings
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_docs = openai_embeddings.embed_documents(["doc chunk 1", "doc chunk 2"])
```
### What They Do
*   `HuggingFaceEmbeddings`: Generates vector representations locally ($0 API cost) using open-source models like `all-MiniLM-L6-v2`. Use `model_kwargs={'device': 'cuda'}` for GPU acceleration.
*   `OpenAIEmbeddings`: Generates high-quality semantic vectors via the OpenAI API. Use the newer `text-embedding-3-small` model (cheaper and supports custom output dimension reduction).
*   **Key Methods**:
    *   `embed_documents(list_of_texts)`: Embeds multiple document chunks (indexing phase).
    *   `embed_query(single_text)`: Embeds the user query (search phase).
*   **Key Concept**: Cosine similarity measures the angle between vectors to check document similarity, bypassing issues with document length variance.

---

## 8. Vector Databases (`8.1` - `8.4`)
### Imports
```python
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
```
### How to Use
```python
# 1. FAISS (Local in-memory index)
db_faiss = FAISS.from_documents(chunks, openai_embeddings)
db_faiss.save_local("faiss_index")
loaded_faiss = FAISS.load_local("faiss_index", openai_embeddings, allow_dangerous_deserialization=True)

# 2. Chroma (Local DB persistent to disk)
db_chroma = Chroma.from_documents(chunks, openai_embeddings, persist_directory="./chroma_db")

# 3. Pinecone (Managed cloud index)
pc = Pinecone(api_key="your_api_key")
db_pinecone = PineconeVectorStore(index=pc.Index("rag-index"), embedding=openai_embeddings)
```
### What They Do & When to Use
*   `Chroma`: Local, lightweight open-source vector store. Persistent to disk via SQLite. Excellent for prototyping.
*   `FAISS` (Facebook AI Similarity Search): Blazing fast, pure in-memory local vector index. No database CRUD features; save/load via `save_local()` and `load_local()`.
*   `InMemoryVectorStore`: Simplest, zero-dependency python-dictionary-based store in `langchain-core`. Only for testing/mocking.
*   `PineconeVectorStore`: Managed cloud database. Scale-ready, serverless, and ideal for multi-tenant applications using metadata pre-filters.

#### Understanding Similarity Scores
The similarity score represents how closely related a document chunk is to your query. The scoring depends on the distance metric used:

ChromaDB default: Uses L2 distance (Euclidean distance)

- Lower scores = MORE similar (closer in vector space)
- Score of 0 = identical vectors
- Typical range: 0 to 2 (but can be higher)


Cosine similarity (if configured):

- Higher scores = MORE similar
- Range: -1 to 1 (1 being identical)

---

## 9. RAG Chains & Conversational Memory (`8.1-chromadb.ipynb`)

![RAG Architecture](assets/image-2.png)

### Imports
```python
# LLM Initialization
from langchain_openai import ChatOpenAI
from langchain.chat_models.base import init_chat_model

# Prompts & Message Structure
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Output Parsers & Runnables (LCEL)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Chains & Retrievers (Built-in - langchain_classic for LangChain 1.x compatibility)
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
```

### How to Use

#### 1. Custom RAG Chain using LCEL (LangChain Expression Language)
LCEL provides a declarative way to construct chains using the pipe operator (`|`).

```python
# Initialize LLM
llm = init_chat_model("openai:gpt-4.1-mini") 

# Define a custom prompt template
custom_prompt = ChatPromptTemplate.from_template("""Use the following context to answer the question. 
If you don't know the answer based on the context, say you don't know.
Provide specific details from the context to support your answer.

Context:
{context}
                                                 
Question: {question}

Answer: """)

# Setup retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Helper function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build the custom RAG Chain using LCEL
rag_chain_lcel = (
    {
        "context": retriever | format_docs, 
        "question": RunnablePassthrough()
    }
    | custom_prompt
    | llm
    | StrOutputParser()
)

# Test/Invoke the chain
response = rag_chain_lcel.invoke("What is Deep Learning")

# Query function using the LCEL approach
def query_rag_lcel(question):
    print(f"Question: {question}")
    print("-" * 50)
    
    # Pass string query directly to the chain
    answer = rag_chain_lcel.invoke(question)
    print(f"Answer: {answer}")
    
    # Get source documents separately for inspection
    docs = retriever.invoke(question)
    print("\nSource Documents:")
    for i, doc in enumerate(docs):
        print(f"\n--- Source {i+1} ---")
        print(doc.page_content[:200] + "...")

# Run verification
query_rag_lcel("What are the key concepts in reinforcement learning?")
```

#### 2. Conversational RAG Chain (With History/Memory)
'''  (compare to Modern RAG Chain below only 2 things are different 1. the history-aware retriever and 2. the chat history placeholder in the prompt)'''

To support multi-turn conversation and resolve pronouns (e.g. "what are its uses?"), we use a history-aware retriever.

```python
# 1. Create a prompt for reformulating the question
contextualize_q_system_prompt = """Given a chat history and the latest user question 
which might reference context in the chat history, formulate a standalone question 
which can be understood without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 2. Create the history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# 3. Create a document chain with history
qa_system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# 4. Create conversational RAG chain
conversational_rag_chain = create_retrieval_chain(
    history_aware_retriever, 
    question_answer_chain
)

# 5. Invoke conversational RAG chain
chat_history = []

# First question
result1 = conversational_rag_chain.invoke({
    "chat_history": chat_history,
    "input": "What is machine learning?"
})
print(f"Q: What is machine learning?")
print(f"A: {result1['answer']}")

# Update history
chat_history.extend([
    HumanMessage(content="What is machine learning"),
    AIMessage(content=result1['answer'])
])

# Follow-up question (refers to ML from previous question)
result2 = conversational_rag_chain.invoke({
    "chat_history": chat_history,
    "input": "What are its main types?"
})
print(f"Q: What are its main types?")
print(f"A: {result2['answer']}")
```

#### 3. Modern RAG Chain (Using LangChain Classic Retrieval Chain)
![alt text](image-1.png)
The classic RAG chain uses helper functions like `create_stuff_documents_chain` and `create_retrieval_chain` to quickly stitch together a retriever, prompt, and LLM.

```python
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Configure Environment (Euron API compatibility)
os.environ["OPENAI_API_KEY"] = os.getenv("EURI_API_KEY")
os.environ["OPENAI_BASE_URL"] = "https://api.euron.one/api/v1/euri"

sample_text = "Machine Learning is fascinating"
embeddings = OpenAIEmbeddings()

persist_directory = "./chroma_db"

# Create a ChromaDB vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name="rag_collection"
)

print(f"Vector store named : {vectorstore._collection.name} ")
print(f"Vector store created with {vectorstore._collection.count()} vectors")
print(f"Persisted to: {persist_directory}")

# Setup retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# Setup prompt template
system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Create stuff documents chain & retrieval chain
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# Invoke the RAG chain
response = rag_chain.invoke({"input": "What is Deep Learning"})
```

### What They Do
*   `ChatOpenAI` / `init_chat_model`: Direct instantiation vs a configurable factory helper to initialize chat models.
*   `ChatPromptTemplate`: Creates structured message prompts for the LLM.
*   `StrOutputParser`: Extracts the string content from the LLM's response message object.
*   `RunnablePassthrough`: Passes the input unmodified through the current step (useful for mapping user queries).
*   `create_stuff_documents_chain`: Combines a list of documents into a single prompt template context window.
*   `create_retrieval_chain`: Chains a retriever and stuff-documents chain together.
*   `create_history_aware_retriever`: Combines conversation history and user query, asking the LLM to draft a standalone query *before* searching the Vector DB. Ensures correct pronoun resolution (e.g. "it", "them").

---

## 10. Semantic Chunking (`9.1-semantichunking.ipynb`)
### Imports
```python
from langchain_experimental.text_splitter import SemanticChunker
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
```
### How to Use
```python
# Split documents based on semantic shifts between sentences
semantic_splitter = SemanticChunker(openai_embeddings, breakpoint_threshold_type="percentile")
semantic_chunks = semantic_splitter.split_documents(documents)
```
### What They Do
*   `SemanticChunker`: Splits documents based on semantic content shifts. It calculates embedding similarity between consecutive sentences and creates a boundary when similarity drops below a threshold.
*   **Key Concept**: Far superior to fixed-size chunkers for preserving complete ideas/topics. However, it is computationally heavy.
*   **Best Practice**: Use a lightweight, free local embedding model (like `all-MiniLM-L6-v2`) for the chunking calculation phase, and save API embeddings for final index storage.
