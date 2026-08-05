# RAG (Retrieval-Augmented Generation) Comprehensive Notes (1-13)

---

## 📋 Table of Contents

1. [Data Ingestion & Splitting](#1-data-ingestion--splitting-1-dataingestionipynb)

2. [PDF Parsing](#2-pdf-parsing-2-dataparsingpdfipynb)

3. [Word Document Parsing](#3-word-document-parsing-3-dataparsingdocipynb)

4. [CSV & Excel Structured Parsing](#4-csv--excel-structured-parsing-4-csvexcelparsingipynb)

5. [JSON Parsing](#5-json-parsing-5-jsonparsingipynb)

6. [Database Parsing](#6-database-parsing-6-databaseparsingipynb)

7. [Embedding Models](#7-embedding-models-70-embeddingipynb--71-openaiembeddingsipynb)

8. [Vector Databases](#8-vector-databases-81---84)
   * [ChromaDB (`langchain_chroma.Chroma`)](#1-chroma-langchain_chromachroma)
   * [FAISS (`langchain_community.vectorstores.FAISS`)](#2-faiss-langchain_communityvectorstoresfaiss)
   * [Pinecone (`langchain_pinecone.PineconeVectorStore`)](#3-pinecone-langchain_pineconepineconevectorstore)
   * [InMemoryVectorStore](#4-inmemoryvectorstore-langchain_corevectorstoresinmemoryvectorstore)
   * [Vector Distance Metrics & Similarity Scores](#understanding-vector-distance-metrics--similarity-scores)

9. [RAG Chains & Conversational Memory](#9-rag-chains--conversational-memory-81-chromadbipynb)

10. [Semantic Chunking](#10-semantic-chunking-91-semantichunkingipynb)
    * [RAG Chain Types Comparison](#rag-chain-types-comparison)
    * [Vector Store vs Vector Database](#vector-store-vs-vector-database)

11. [Hybrid Search & Re-ranking](#11-hybrid-search--re-ranking-05_hybrid-search)
    * [11.1 Hybrid Retriever – Dense & Sparse Combination](#111-hybrid-retriever--dense--sparse-combination-1-densesparseipynb)
    * [11.2 Re-ranking Hybrid Search Strategies](#112-re-ranking-hybrid-search-strategies-2-reranking-1ipynb)
    * [11.3 Maximal Marginal Relevance - MMR](#113-maximal-marginal-relevance---mmr-3-mmripynb)
    * [11.4 RAG Search Strategies & Production Search Pipelines](#114-rag-search-strategies--production-search-pipelines)

12. [Query Enhancement & Advanced RAG](#12-query-enhancement--advanced-rag-06_query_enhancment)
    * [12.1 Query Expansion](#121-query-expansion-1-queryexpansionipynb)
    * [12.2 Query Decomposition](#122-query-decomposition-2-querydecompositionipynb)
    * [12.3 Hypothetical Document Embeddings (HyDE)](#123-hypothetical-document-embeddings---hyde-3-hydeipynb)

13. [Multimodal RAG](#13-multimodal-rag-07_multimodle-rag)

---

## 1. Data Ingestion & Splitting (`1-dataingestion.ipynb`)
### Imports
```python
import tempfile
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
```

<details>
<summary><b>💡 Helper Pattern: Programmatically Generating Temporary Sample Files</b></summary>

When testing loaders (like `DirectoryLoader`) without cluttering your project folder with hardcoded files, you can create temporary files using Python's `tempfile` module:

```python
import tempfile

# 1. Create temporary directory
temp_dir = tempfile.mkdtemp()

sample_docs = [
    "Artificial Intelligence (AI) is transforming industries.",
    "Machine Learning is a subset of AI focusing on data and algorithms.",
    "Retrieval-Augmented Generation (RAG) enhances LLMs with dynamic external retrieval."
]

# 2. Write sample documents to temporary files
for i, doc in enumerate(sample_docs):
    with open(f"{temp_dir}/doc_{i}.txt", "w") as f:
        f.write(doc)

print(f"Sample documents created in: {temp_dir}")
```
*   **Why use this?** Ideal for unit tests, notebooks, and quick experiments because it creates a isolated sandbox directory that can be safely discarded later.
</details>

### How to Use

#### Single File Loading (`TextLoader`)
```python
# Load single text file and split recursively
loader = TextLoader("data/sample.txt")
documents = loader.load()

# 🔹 KEY CONFIG: chunk_size=500, chunk_overlap=50 (10% overlap ratio rule of thumb)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]  # Preserves paragraph & sentence structures
)
chunks = text_splitter.split_documents(documents)
```

#### Bulk Folder Ingestion (`DirectoryLoader`)
```python
# Bulk load all matching text files from a directory path
dir_loader = DirectoryLoader(
    path=temp_dir,          # Directory path containing documents (e.g., "data/" or temp_dir)
    glob="*.txt",           # File pattern matching
    loader_cls=TextLoader,  # Underlying loader class used for each file
    show_progress=True      # Show progress bar during bulk loading
)
bulk_documents = dir_loader.load()
```

### What They Do
*   `Document`: LangChain's base class storing raw text (`page_content`) and arbitrary key-value dict metadata (`metadata`).
*   `DirectoryLoader`: Scans a directory for matching files (`glob="*.txt"`) and loads them concurrently using a specified single-file loader (`loader_cls=TextLoader`).
*   `RecursiveCharacterTextSplitter`: <mark style="background-color: #d4edda; color: #155724; padding: 2px 4px; border-radius: 4px;">Best default splitter</mark>. Recursively splits text using a hierarchy of separators (`\n\n`, `\n`, `" "`, `""`) to keep paragraphs and sentences visually intact.
*   `CharacterTextSplitter`: Splits text rigidly based on a single character separator (e.g. `\n\n`), risking oversized chunks if separators are far apart.
*   `TokenTextSplitter`: Splits text strictly by token count (using OpenAI `tiktoken`) to guarantee chunk sizes fit exact LLM context boundaries.
*   `TextLoader` / `DirectoryLoader`: Loads a single plain text file / bulk-loads matching document files from a folder directory.

### 💡 Advanced Best Practices & Key Insights:
*   **Chunk Overlap Strategy**: Always set `chunk_overlap` between <mark style="background-color: #fff3cd; color: #856404; padding: 2px 4px; border-radius: 4px;">10% to 20% of `chunk_size`</mark>. This prevents losing critical semantic context across chunk boundary splits.
*   **Metadata Enrichment**: Always inject custom metadata attributes (e.g., `source`, `creation_date`, `category`) onto each Document during ingestion for precise metadata filtering in vector databases.

<br>

---

<br>

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
*   `PyMuPDFLoader`: C-based, high-accuracy PDF text extractor. <mark style="background-color: #d4edda; color: #155724; padding: 2px 4px; border-radius: 4px;">Extremely fast (10x faster than PyPDF)</mark> and superior at multi-column layout parsing.

### 💡 Advanced Best Practices & Key Insights:
*   **Ligature & Whitespace Cleaning**: Raw PDF text often contains ligatures (`ﬁ`, `ﬂ`) or whitespace artifacts. Always run regex cleaning (`re.sub(r'\s+', ' ', text)`) post-ingestion.
*   **Scanned PDFs**: For image-based PDFs, standard loaders fail. Use OCR tools like `pdf2image` + `pytesseract` or `UnstructuredPDFLoader` with OCR strategy.

<br>

---

<br>

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
unstructured_loader = UnstructuredWordDocumentLoader(
    "data/sample.docx", 
    mode="elements", 
    strategy="fast"
)
element_docs = unstructured_loader.load()
```
### What They Do
*   `Docx2txtLoader`: Fast, lightweight loader that extracts plain text from `.docx` files.
*   `UnstructuredWordDocumentLoader`: Partitions documents into granular logical elements (`Title`, `NarrativeText`, `Table`).

### 💡 Advanced Best Practices & Key Insights:
*   **Header/Footer Exclusion**: Use `UnstructuredWordDocumentLoader` with `mode="elements"` to filter out repeating header/footer noise from indexing.

<br>

---

<br>

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
*   `pandas.DataFrame`: Used for custom CSV/Excel pre-processing before converting to LangChain Documents.

### 💡 Advanced Best Practices & Key Insights:
*   **Tabular Anti-Pattern**: Avoid indexing huge tables row-by-row into vector stores. Instead, format rows into rich natural language summaries (`"Product X belongs to Category Y with price $Z"`) for drastically better semantic retrieval.

<br>

---

<br>

## 5. JSON Parsing (`5-jsonparsing.ipynb`)
### Imports
```python
from langchain_community.document_loaders import JSONLoader
```
### How to Use
```python
# Load specific values from nested JSON using jq path queries
loader = JSONLoader(
    "data/company.json", 
    jq_schema=".employees[].role", 
    text_content=True
)
docs = loader.load()
```
### What They Do
*   `JSONLoader`: Extracts specific JSON elements using `jq` path syntax (`.employees[].role`).

### 💡 Advanced Best Practices & Key Insights:
*   **JSON Noise Reduction**: Embed only human-readable descriptive values (`text_content=True`). Store structural JSON metadata (IDs, foreign keys) in the Document `metadata` dict for exact filtering.

<br>

---

<br>

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
*   `SQLDatabase`: Connects to relational databases (SQLite, PostgreSQL, MySQL) and exposes DDL schema metadata.
*   `SQLDatabaseLoader`: Executes queries and maps SQL result rows into LangChain Document objects.

### 💡 Advanced Best Practices & Key Insights:
*   **Security & Read-Only Access**: Never connect a Text-to-SQL or database ingestion agent using write/delete database credentials. Always scope SQL users to strictly <mark style="background-color: #f8d7da; color: #721c24; padding: 2px 4px; border-radius: 4px;">READ-ONLY</mark> permissions.

<br>

---

<br>

## 7. Embedding Models (`7.0-embedding.ipynb` & `7.1-openaiembeddings.ipynb`)
### 🔑 Setting up API Keys for Cloud Embedding Models (e.g., OpenAI)

To use API-based embedding models like `OpenAIEmbeddings`, you need to set up your API key. There are two primary methods:

#### Method 1: Using `.env` File (Recommended Best Practice)
1. Install `python-dotenv`:
   ```bash
   pip install python-dotenv
   ```
2. Create a `.env` file in your root project directory:
   ```env
   OPENAI_API_KEY="your-actual-openai-api-key-here"
   ```
3. Load the environment variable in Python:
   ```python
   import os
   from dotenv import load_dotenv

   load_dotenv()  # Automatically loads OPENAI_API_KEY into os.environ
   ```

#### Method 2: Passing directly in Constructor
```python
import os
from langchain_openai import OpenAIEmbeddings

openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key="your-actual-openai-api-key-here" # Or os.getenv("OPENAI_API_KEY")
)
```

### Imports
```python
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

load_dotenv()
```
### How to Use
```python
# Local HuggingFace Embeddings (No API Key Required - Runs Locally)
hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_query = hf_embeddings.embed_query("your query text")

# API-Based OpenAI Embeddings (Requires OPENAI_API_KEY)
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_docs = openai_embeddings.embed_documents(["doc chunk 1", "doc chunk 2"])
```
### What They Do
*   `HuggingFaceEmbeddings`: Generates vector representations locally ($0 API cost, no API Key needed) using open-source models like `all-MiniLM-L6-v2`. Use `model_kwargs={'device': 'cuda'}` for GPU acceleration.
*   `OpenAIEmbeddings`: Generates high-quality semantic vectors via the OpenAI API (Requires `OPENAI_API_KEY`). Use the newer `text-embedding-3-small` model (cheaper and supports custom output dimension reduction).
*   **Key Methods**:
    *   `embed_documents(list_of_texts)`: Embeds multiple document chunks (indexing phase).
    *   `embed_query(single_text)`: Embeds the user query (search phase).
*   **Key Concept**: Cosine similarity measures the angle between vectors to check document similarity, bypassing issues with document length variance.

<br>

---

<br>

## 8. Vector Databases (`8.1` - `8.4`)

### Overview & Comparison
Vector databases store and index high-dimensional vector embeddings generated by machine learning models to perform fast nearest-neighbor similarity searches (like Cosine Similarity or Euclidean L2 Distance).

| Vector DB | Type | Storage / Persistence | Key Advantage | Best Use Case |
|-----------|------|-----------------------|---------------|---------------|
| **ChromaDB** | <mark style="background-color: #fff3cd; color: #856404; padding: 2px 4px; border-radius: 4px;">Open Source / Local & Cloud</mark> | Disk (SQLite/Parquet) or Cloud Server / Hosted Cloud | <mark style="background-color: #e2e3e5; color: #383d41; padding: 2px 4px; border-radius: 4px;">Zero-config local setup</mark>, persistence, client/server & managed cloud support | <mark style="background-color: #d1ecf1; color: #0c5460; padding: 2px 4px; border-radius: 4px;">Local RAG & rapid prototyping</mark>, microservices via Chroma Cloud |
| **FAISS** | <mark style="background-color: #fff3cd; color: #856404; padding: 2px 4px; border-radius: 4px;">In-Memory C++ Library</mark> | Local Files (`.index` / `.pkl`) | <mark style="background-color: #d4edda; color: #155724; padding: 2px 4px; border-radius: 4px;">Blazing fast GPU/CPU vector search</mark> | <mark style="background-color: #d1ecf1; color: #0c5460; padding: 2px 4px; border-radius: 4px;">In-memory batch vector searches</mark>, localized search without server overhead |
| **Pinecone** | <mark style="background-color: #f8d7da; color: #721c24; padding: 2px 4px; border-radius: 4px;">Cloud Managed Service</mark> | Fully Managed Serverless Cloud | <mark style="background-color: #d4edda; color: #155724; padding: 2px 4px; border-radius: 4px;">Zero infra management</mark>, enterprise scaling & real-time updates | <mark style="background-color: #d1ecf1; color: #0c5460; padding: 2px 4px; border-radius: 4px;">Production RAG pipelines</mark>, multi-tenant SaaS applications |
| **AstraDB / DataStax** | <mark style="background-color: #f8d7da; color: #721c24; padding: 2px 4px; border-radius: 4px;">Cloud Managed Service</mark> | Serverless Cassandra | <mark style="background-color: #e2e3e5; color: #383d41; padding: 2px 4px; border-radius: 4px;">Vector + NoSQL JSON docs</mark>, global scale & low latency | <mark style="background-color: #d1ecf1; color: #0c5460; padding: 2px 4px; border-radius: 4px;">Enterprise hybrid data RAG</mark> requiring document storage |
| **InMemoryVectorStore** | <mark style="background-color: #fff3cd; color: #856404; padding: 2px 4px; border-radius: 4px;">Core Store</mark> | In-Memory Python Dict | <mark style="background-color: #e2e3e5; color: #383d41; padding: 2px 4px; border-radius: 4px;">Zero external dependencies</mark>, pure Python execution | <mark style="background-color: #d1ecf1; color: #0c5460; padding: 2px 4px; border-radius: 4px;">Unit testing & CI/CD</mark>, single-session demo scripts |

---

### 💡 When to Use Which Vector Database? (Decision Guide)

1. **Use <mark style="background-color: #fff3cd; color: #856404; padding: 2px 6px; border-radius: 4px;">ChromaDB</mark> when:**
   * You are building **local RAG applications**, Python scripts, or desktop tools.
   * You want an easy, lightweight database that runs embedded in Python or in Docker containers.
   * You plan to transition from local testing to a remote microservice or managed **<mark style="background-color: #d1ecf1; color: #0c5460; padding: 2px 4px; border-radius: 4px;">Chroma Cloud</mark>** (`HttpClient` / Hosted Service) without changing your application query logic.

2. **Use <mark style="background-color: #d4edda; color: #155724; padding: 2px 6px; border-radius: 4px;">FAISS</mark> when:**
   * You need **<mark style="background-color: #fff3cd; color: #856404; padding: 2px 4px; border-radius: 4px;">maximum similarity search speed</mark>** over fixed/static vector datasets.
   * You want to run searches purely in memory or perform high-throughput **GPU-accelerated** indexing.
   * You do **not** need multi-user concurrency, API server features, or real-time CRUD operations.

3. **Use <mark style="background-color: #f8d7da; color: #721c24; padding: 2px 6px; border-radius: 4px;">Pinecone</mark> when:**
   * You are deploying **<mark style="background-color: #d4edda; color: #155724; padding: 2px 4px; border-radius: 4px;">production RAG applications</mark>** to cloud environments.
   * You require a **fully managed serverless infrastructure** with automatic scaling, backup, high availability, and metadata filtering.
   * You want zero infrastructure maintenance (no managing servers or disk storage).

4. **Use <mark style="background-color: #e2e3e5; color: #383d41; padding: 2px 6px; border-radius: 4px;">DataStax / AstraDB</mark> when:**
   * You require enterprise-grade serverless cloud vector search backed by Apache Cassandra.
   * You need both **rich JSON document/NoSQL storage** alongside vector embeddings.

5. **Use <mark style="background-color: #fff3cd; color: #856404; padding: 2px 6px; border-radius: 4px;">InMemoryVectorStore</mark> when:**
   * You are running **unit tests**, continuous integration (CI) tests, or quick 1-file proof-of-concepts where vectors don't need to persist after Python exits.

---

### 1. Chroma (`langchain_chroma.Chroma`)
Chroma is an open-source, developer-friendly vector database. It supports 3 deployment modes:
* 🟢 **Embedded Mode**: Runs inside your Python process and persists data to disk via SQLite/Parquet (ideal for local development).
* 🟡 **Client/Server Mode**: Runs Chroma as a standalone Docker container or separate microservice (`HttpClient`).
* 🔵 **Chroma Cloud**: Fully managed serverless cloud service (`chromadb.CloudClient`) without local infrastructure overhead.

#### Imports
```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
```

#### Complete Implementation (Create, Persist, Load & Query)
```python
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

class ChromaVectorStoreManager:
    def __init__(self, persitent_dir="./chroma_db"):
        self.persitent_dir = persitent_dir
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-small") # 🔹 Standardize embedding model

    def create_and_persist_vector_store(self, chunks):
        """
        Creates a Chroma vector store from document chunks and automatically persists it to disk.
        """
        # 🔹 from_documents() builds vector index and writes directly to disk
        db = Chroma.from_documents(
            chunks,
            self.embedding,
            persist_directory=self.persitent_dir
        )
        return db

    def load_vector_store(self):
        """
        Loads an existing persisted Chroma vector store from disk.
        """
        return Chroma(
            persist_directory=self.persitent_dir,
            embedding_function=self.embedding
        )

# ⚡ Execution Flow:
# 1. Create store  -> manager.create_and_persist_vector_store(chunks)
# 2. Reload store  -> loaded_db = manager.load_vector_store()
# 3. Vector search -> loaded_db.similarity_search("What is RAG?", k=3)
```

> [!NOTE]
> **Is `db.persist()` still needed?**
> **No.** `db.persist()` is **deprecated and removed** in `chromadb` (v0.4.0+) & `langchain-chroma`. Data is saved automatically whenever `persist_directory` is specified. Calling `db.persist()` will throw an `AttributeError`.

#### ➕ Adding Data to Existing Vector Store (Incremental Ingestion)
To add new documents or raw text to an existing collection without rebuilding the database:

```python
from langchain_core.documents import Document

# 1. Load existing vector store
db = manager.load_vector_store()

# Option A: Adding LangChain Document objects (with metadata)
new_docs = [
    Document(page_content="New chunk content 1", metadata={"source": "news_api", "category": "tech"}),
    Document(page_content="New chunk content 2", metadata={"source": "news_api", "category": "finance"})
]
added_ids = db.add_documents(new_docs)  # Embeds and auto-persists

# Option B: Adding raw strings directly
added_ids = db.add_texts(
    texts=["Raw string 1 to embed", "Raw string 2 to embed"],
    metadatas=[{"author": "Alice"}, {"author": "Bob"}]
)
```

#### Metadata Pre-filtering & CRUD Operations
```python
# 1. Metadata Pre-Filtering Search
filtered_docs = db.similarity_search(
    "What is deep learning?",
    k=3,
    filter={"source": "data/sample.pdf", "page": 1}
)

# 2. Collection Inspection (Total Count)
print(f"Total stored vectors: {db._collection.count()}")

# 3. Delete Documents by ID
db.delete(ids=["doc_id_1", "doc_id_2"])
```

#### Core Methods Summary
* `Chroma.from_documents()`: Embeds chunks and auto-saves indexed vectors to `persist_directory`.
* `Chroma()`: Loads an existing persisted store from disk without re-embedding.
* `db.add_documents()` / `db.add_texts()`: Incrementally adds and auto-saves new data.
* `db.similarity_search()`: Performs semantic nearest-neighbor similarity search.

---

### 2. FAISS (`langchain_community.vectorstores.FAISS`)
FAISS (Facebook AI Similarity Search) is a high-performance C++ library with Python bindings designed for fast similarity search and vector clustering.

#### Imports
```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
```

#### Complete Implementation (Create, Save & Load)
```python
class FAISSVectorStoreManager:
    def __init__(self, persitent_dir="faiss_index"):
        self.persitent_dir = persitent_dir
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    def create_and_persist_vector_store(self, chunks):
        """
        Creates an in-memory FAISS index and writes files (.faiss + .pkl) to disk.
        """
        db = FAISS.from_documents(chunks, self.embedding)
        
        # 🔹 MANDATORY: Save binary vector index & metadata docstore to disk
        db.save_local(folder_path=self.persitent_dir)
        return db

    def load_vector_store(self):
        """
        Loads a saved FAISS index from disk into memory.
        """
        return FAISS.load_local(
            folder_path=self.persitent_dir,
            embeddings=self.embedding,
            allow_dangerous_deserialization=True  # Required to safely unpickle docstore
        )
```

> [!WARNING]
> **Does Auto-Save happen in FAISS like Chroma?**
> **NO! FAISS is purely IN-MEMORY.**
> - Chroma auto-writes to SQLite on disk, while FAISS holds vectors strictly in Python RAM.
> - **Mandatory**: You **MUST** call `db.save_local(folder_path)` after creating or updating (`add_documents()`) a FAISS store to persist changes.
> - Exiting Python without calling `save_local()` will lose all newly added vectors!

#### ➕ Adding Data & Saving (FAISS Incremental Workflow)
```python
# 1. Load FAISS store into memory
db = manager.load_vector_store()

# 2. Add new documents into RAM
db.add_documents(new_docs)

# 3. Save updated RAM state back to disk
db.save_local(folder_path="faiss_index")
```

#### Core Methods Summary
* `FAISS.from_documents()`: Constructs an in-memory vector index.
* `db.save_local()`: Serializes and saves index binary (`index.faiss`) & metadata (`index.pkl`) to disk.
* `FAISS.load_local()`: Loads saved index files back into RAM.
* `db.add_documents()`: Appends new vectors to RAM (must be followed by `db.save_local()`).

---

### 3. Pinecone (`langchain_pinecone.PineconeVectorStore`)
Pinecone is a cloud-native, fully-managed serverless vector database designed for production scaling, high availability, and metadata pre-filtering.

#### Imports
```python
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
```

#### Complete Implementation (Create & Load)
```python
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

class PineconeVectorStoreManager:
    def __init__(self, index_name="rag-index"):
        self.index_name = index_name
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    def create_and_persist_vector_store(self, chunks):
        """
        Creates index if missing and embeds chunks directly into cloud Pinecone index.
        """
        existing_indexes = [idx["name"] for idx in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # Vector dimensions for text-embedding-3-small
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            
        # Create and persist chunks to cloud vector store
        db = PineconeVectorStore.from_documents(
            chunks,
            self.embedding,
            index_name=self.index_name
        )
        return db

    def load_vector_store(self):
        """
        Connects to an existing cloud-hosted Pinecone vector store index.
        """
        return PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embedding
        )
```

---

### 4. InMemoryVectorStore (`langchain_core.vectorstores.InMemoryVectorStore`)
`InMemoryVectorStore` is the simplest zero-dependency transient vector store provided natively by `langchain-core` (LangChain 0.2+ standard). It stores vectors in a plain Python dictionary in memory (RAM).

#### Imports
```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
```

#### Basic Usage (In-Memory Execution)
```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create store in RAM
db = InMemoryVectorStore.from_documents(chunks, embeddings)

# Query store
results = db.similarity_search("Explain vector embeddings", k=2)
```

#### 💾 Can you Save & Load `InMemoryVectorStore` to/from Disk?
By default, `InMemoryVectorStore` data is lost when Python exits. However, because it is a pure Python object, **you CAN serialize it to disk** (save) and reload it later using Python's `pickle` or custom JSON dumping!

##### Save & Load using `pickle` (File Dump):
```python
import pickle

# 1. Save (Dump) in-memory vector store to disk
with open("in_memory_store.pkl", "wb") as f:
    pickle.dump(db, f)

# 2. Load (Restore) in-memory vector store from disk
with open("in_memory_store.pkl", "rb") as f:
    loaded_db = pickle.load(f)

# Query loaded store
results = loaded_db.similarity_search("What is RAG?", k=2)
```

> [!TIP]
> **When to use `InMemoryVectorStore`?**
> Ideal for **unit testing (CI/CD)**, short interactive demo scripts, or single session apps where you don't want external database binaries installed on your system.

---

### Understanding Vector Distance Metrics & Similarity Scores

Similarity search relies on mathematical distance metrics between high-dimensional vector embeddings:

1. **Cosine Similarity**:
   * Measures the angle cosine between two vectors.
   * **Range**: `-1.0` to `1.0` (or normalized `0.0` to `1.0`).
   * **Interpretation**: **Higher is MORE similar**. `1.0` represents identical vector direction regardless of text length.

2. **Euclidean Distance (L2 Distance)**:
   * Measures straight-line geometric distance between vector points in multi-dimensional space.
   * **Range**: `0.0` to `+∞`.
   * **Interpretation**: **Lower is MORE similar**. `0.0` represents identical vectors.

3. **Dot Product (Inner Product)**:
   * Measures both vector angle and magnitude. Fast for normalized vectors where dot product equals cosine similarity.

> **Important Note on Score Sorting**:
> * `db.similarity_search(query)` returns Documents ordered by relevance.
> * `db.similarity_search_with_score(query)` returns tuples `(Document, score)`. When using L2 distance metrics (e.g., Chroma default), **smaller scores represent closer matches**. When using cosine similarity, **larger scores represent closer matches** prints.

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

#### 1. LLM / Model Initialization Methods

**Method A: Using LangChain's Factory Function**
```python
# Uses environment variables (OPENAI_API_KEY, OPENAI_BASE_URL) automatically
llm = init_chat_model("openai:gpt-4.1-mini") 
```

**Method B: Making a Model Using Native Client (OpenAI API directly)**
If you want to bypass LangChain and interact with the LLM directly using a client:
```python
from openai import OpenAI

# Initialize the client
client = OpenAI(
    api_key="your_api_key",
    base_url="https://api.euron.one/api/v1/euri" # Optional: for custom providers
)

# Make a model request using the client
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "What is Deep Learning?"}]
)
print(response.choices[0].message.content)
```

#### 2. Custom RAG Chain using LCEL (LangChain Expression Language)
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

#### 3. Conversational RAG Chain (With History/Memory)
# Document Retrival & Chain Construction
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Local Vector Database & Embeddings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
```

#### How to Use (Conversational RAG with Chat History)
```python
# 1. Setup Vector Store Retriever & LLM
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# 2. Contextualize Question Prompt
contextualize_q_system_prompt = """Given a chat history and the latest user question 
which might reference context in the chat history, formulate a standalone question 
which can be understood without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 3. Create History-Aware Retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# 4. Answer Generation Prompt
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

# 5. Build Complete Conversational RAG Chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
conversational_rag_chain = create_retrieval_chain(
    history_aware_retriever, 
    question_answer_chain
)
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

#### 4. Modern RAG Chain (Using LangChain Classic Retrieval Chain)
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
# Pure Python / SentenceTransformers implementation
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# LangChain & RAG Pipeline Integration
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableMap
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
```

### How to Use

#### Method A: Custom Threshold-Based Semantic Chunker (Pure Python / SentenceTransformers)
```python
class ThresholdSemanticChunker:
    def __init__(self, model_name="all-MiniLM-L6-v2", threshold=0.7):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold 

    def split(self, text: str):
        # 1. Split text into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return []
        
        # 2. Compute embeddings for all sentences
        embeddings = self.model.encode(sentences)
        chunks = []
        current_chunk = [sentences[0]]

        # 3. Calculate cosine similarity between consecutive adjacent sentences
        for i in range(1, len(sentences)):
            sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
            if sim >= self.threshold:
                current_chunk.append(sentences[i])
            else:
                chunks.append(". ".join(current_chunk) + ".")
                current_chunk = [sentences[i]]

        chunks.append(". ".join(current_chunk) + ".")
        return chunks

    def split_documents(self, docs):
        result = []
        for doc in docs:
            for chunk in self.split(doc.page_content):
                result.append(Document(page_content=chunk, metadata=doc.metadata))
        return result

# Usage
chunker = ThresholdSemanticChunker(threshold=0.7)
chunks = chunker.split_documents([Document(page_content="...")])
```

#### Method B: Built-in LangChain `SemanticChunker`
```python
# 1. Load Document
loader = TextLoader("langchain_intro.txt")
docs = loader.load()

# 2. Initialize embedding model (e.g., OpenAI or local HuggingFace)
embeddings = OpenAIEmbeddings()

# 3. Initialize Semantic Chunker (default breakpoint threshold type: percentile)
chunker = SemanticChunker(
    embeddings, 
    breakpoint_threshold_type="percentile" # Options: 'percentile', 'standard_deviation', 'interquartile', 'gradient'
)

# 4. Split documents into semantically coherent chunks
semantic_chunks = chunker.split_documents(docs)
```

#### Method C: Modular RAG Pipeline with Semantic Chunker & FAISS
```python
# 1. Custom semantic chunking
chunker = ThresholdSemanticChunker(threshold=0.7)
chunks = chunker.split_documents(docs)

# 2. Store in FAISS Vector Store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# 3. Prompt & LLM Setup
prompt = PromptTemplate.from_template(
    "Answer the question based on the following context:\n\n{context}\n\nQuestion: {question}\n"
)
llm = init_chat_model(model="groq:gemma2-9b-it", temperature=0.4)

# 4. Build LCEL Chain
rag_chain = (
    RunnableMap({
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"],
    })
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Run Query
result = rag_chain.invoke({"question": "What is LangChain used for?"})
print(result)
```

### What They Do
*   `SemanticChunker`: Splits documents dynamically based on sentence embedding similarity rather than arbitrary length-based character limits.
*   `cosine_similarity`: Measures cosine similarity between consecutive sentence vectors. If similarity falls below `threshold`, a topic shift is detected and a chunk boundary is placed.
*   `breakpoint_threshold_type`: Strategy used by LangChain to calculate split thresholds:
    *   `percentile` (default): Splits when sentence distance exceeds a specified percentile cutoff across the document.
    *   `standard_deviation`: Splits based on standard deviation distance from the mean similarity score.
    *   `interquartile`: Uses interquartile range (IQR) to detect statistical anomalies/topic shifts.
    *   `gradient`: Looks for peaks in cosine distance gradients across sentences.

### 💡 Interview & Learning Notes

#### **Key Interview Questions:**
1. **Why use Semantic Chunking instead of `RecursiveCharacterTextSplitter`?**
   * Fixed-size chunkers (like character/token splitters) often cut text in the middle of a paragraph or thought, separating pronouns from their subjects.
   * Semantic chunking calculates similarity between adjacent sentences and only splits when similarity drops below a threshold (signaling a topic change). This preserves semantic context and leads to higher RAG retrieval quality.

2. **What is the main downside of Semantic Chunking?**
   * **Cost & Time**: Requires running an embedding model on *every single sentence* during the ingestion phase before final chunks are created. Using commercial API embeddings (e.g., OpenAI) for chunking can be expensive and slow.

#### **Learning Takeaways:**
* Use a fast, free local embedding model (e.g., `all-MiniLM-L6-v2`) for the sentence-level chunking pass, and save API embeddings (e.g., OpenAI) for indexing into the final vector database.

### 🚀 Best Practices for Semantic Chunking

1. **Cost & Token Optimization**:
   * Never use expensive API models (`text-embedding-3-large`) inside `SemanticChunker`. Use a fast local model (`HuggingFaceEmbeddings` / `SentenceTransformer`) for chunking logic, and use OpenAI embeddings only during final `vectorstore` indexing.

2. **Speed & Scalability**:
   * Semantic chunking is computationally heavy. For massive datasets (millions of documents), use standard `RecursiveCharacterTextSplitter` unless context fragmentation is demonstrably degrading answer quality.

3. **Threshold Tuning**:
   * The `percentile` breakpoint threshold metric is typically the most robust across varying document types and lengths.

---

## RAG Chain Types Comparison

| Chain Type | Description | Key Features | Typical Use Cases |
|------------|-------------|--------------|-------------------|
| **Normal RAG Chain** | Retrieve a fixed set of top‑k documents, concatenate them, and pass the whole text to the LLM in a single prompt. | • Simple to implement<br>• No state across turns<br>• Limited by LLM context window | • One‑off Q&A<br>• Fact‑lookup where the answer fits in a single request |
| **Conversational RAG Chain** | Extends the normal chain with a memory component (chat history) and a history‑aware retriever that rewrites the user query using the prior conversation. | • Maintains multi‑turn context<br>• Resolves pronouns and references<br>• Uses `create_history_aware_retriever` and `MessagesPlaceholder` | • Customer‑support bots<br>• Interactive tutoring with follow‑up questions |
| **Streaming RAG Chain** | Retrieves documents incrementally and streams them to the LLM as they become available (e.g., using LangChain’s Runnable streaming or async generators). | • Handles very large corpora beyond the LLM token limit<br>• Overlaps retrieval and generation to reduce latency<br>• Can provide partial answers early | • Long‑form summarisation<br>• Real‑time assistance over massive knowledge bases |

---

## Vector Store vs Vector Database

| Type | Persistence | Scaling | Metadata / Filtering | Typical Scenarios |
|------|--------------|---------|----------------------|-------------------|
| **In‑Memory Vector Store** (`InMemoryVectorStore`) | Pure Python dict, lives only while the process runs | Limited to a single process, RAM‑bound | Minimal – usually only the vector itself | Unit tests, quick prototypes, notebooks |
| **Local Vector Store** (`FAISS`, `Chroma`) | Files on disk (`.index`, SQLite for Chroma) | Works on a single machine; can handle millions of vectors with appropriate hardware | Supports basic metadata filters (e.g., `metadata['source'] == 'pdf'`) | Personal projects, research notebooks, small‑scale apps |
| **Managed Vector Database** (`Pinecone`, `Weaviate`, `Qdrant`, `Milvus Cloud`) | Cloud‑managed storage with replication & backups | Horizontally scalable, multi‑region, handles billions of vectors | Rich metadata queries, hybrid search (vector + scalar), security controls | Production SaaS products, multi‑tenant services, real‑time recommendation engines |

*When to choose which:* 
- **Prototype / Experiment** → start with **FAISS** or **Chroma** for speed and zero‑cost.
- **Production with modest load** → **Chroma** persisted locally or a lightweight **Qdrant** instance.
- **Enterprise‑grade, high‑throughput** → a managed service like **Pinecone** or **Weaviate** that offers SLA, automatic scaling, and fine‑grained metadata filtering.


---

## 11. Hybrid Search & Re-ranking (`05_hybrid search`)

### 11.1 Hybrid Retriever – Dense & Sparse Combination (`1-densesparse.ipynb`)
#### Imports
```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
```
#### How to Use
```python
# 1. Sample documents & Dense Retriever (FAISS + HuggingFace)
docs = [
    Document(page_content="LangChain helps build LLM applications."),
    Document(page_content="Pinecone is a vector database for semantic search."),
    Document(page_content="LangChain can be used to develop agentic AI applications.")
]
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
dense_vectorstore = FAISS.from_documents(docs, embedding_model)
dense_retriever = dense_vectorstore.as_retriever()

# 2. Sparse Retriever (BM25)
sparse_retriever = BM25Retriever.from_documents(docs)
sparse_retriever.k = 3

# 3. Combine using EnsembleRetriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.7, 0.3]
)

# 4. Invoke hybrid search or connect to RAG chain
results = hybrid_retriever.invoke("How can I build an application using LLMs?")
```
#### What They Do
*   `Dense Retriever`: Uses vector embeddings to capture semantic context, handling synonyms and concept matching.
*   `Sparse Retriever` (`BM25`): Uses keyword-based frequency scoring (TF-IDF family) to catch exact term matches, proper nouns, and technical IDs.
*   `EnsembleRetriever`: Combines search results from multiple retrievers using Reciprocal Rank Fusion (RRF) with user-defined weights (`weights=[0.7, 0.3]`).
*   **Reciprocal Rank Fusion (RRF)**: Re-ranks items by summing reciprocal ranks across retrievers using formula $RRF\_Score(d) = \sum_{m \in M} \frac{w_m}{k + r_m(d)}$ where $r_m(d)$ is the rank of document $d$ in retriever $m$. This ensures documents appearing near the top of *both* dense and sparse searches get the highest final ranking.

---

### 11.2 Re-ranking Hybrid Search Strategies (`2-reranking (1).ipynb`)
#### Imports
```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
```
#### How to Use
```python
# Stage 1: Fast initial retrieval (Fetch top-k=8 candidate docs)
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
retrieved_docs = retriever.invoke("How can I use LangChain with memory?")

# Stage 2: Re-ranking via LLM / Cross-Encoder
prompt = PromptTemplate.from_template("""
You are a helpful assistant. Rank the following documents from most to least relevant to the user's question.

User Question: "{question}"

Documents:
{documents}

Return a list of document indices in ranked order starting from the most relevant.
Output format: comma-separated document indices (e.g., 2,1,4,3)
""")

chain = prompt | llm | StrOutputParser()
formatted_docs = "\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(retrieved_docs)])
response = chain.invoke({"question": query, "documents": formatted_docs})

# Parse indices and re-order documents
indices = [int(x.strip()) - 1 for x in response.split(",") if x.strip().isdigit()]
reranked_docs = [retrieved_docs[i] for i in indices if 0 <= i < len(retrieved_docs)]
```
#### What They Do
*   **Two-Stage Retrieval**: Stage 1 prioritizes **high recall** (fetching a large initial set of candidate chunks quickly). Stage 2 prioritizes **high precision** (using a slower, more powerful model to filter out irrelevant context).
*   **LLM / Cross-Encoder Reranker**: Evaluates joint context between query and documents to eliminate false-positive vector hits before passing final context to the answer generator.

---

### 11.3 Maximal Marginal Relevance - MMR (`3-mmr.ipynb`)
#### Imports
```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
```
#### How to Use
```python
# Create MMR Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.5}
)

# Connect to RAG chain
document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)
response = rag_chain.invoke({"input": "How does LangChain support agents and memory?"})
```
#### What They Do
*   `MMR` (Maximal Marginal Relevance): Optimizes for both **similarity to query** and **diversity among selected documents**.
*   `fetch_k`: Number of candidate documents fetched initially for similarity.
*   `lambda_mult`: Controls trade-off between relevance and diversity (`1.0` = maximum relevance, `0.0` = maximum diversity).
*   **Key Concept**: MMR avoids retrieving multiple near-identical text chunks, maximizing topic coverage within the LLM context window.

<br>

---

<br>

### 11.4 RAG Search Strategies & Production Search Pipelines

Retrieval performance directly governs the accuracy and grounding of a RAG pipeline. Below is a comprehensive breakdown of all major search paradigms used across modern AI systems.

#### 📊 Search Types in Retrieval-Augmented Generation

| Search Type | Description | Best For |
| :--- | :--- | :--- |
| **Keyword Search (Lexical Search)** | Matches exact terms using algorithmic statistical scoring like **BM25** or **TF-IDF**. | Exact domain terms, SKUs, product IDs, code snippets, error messages. |
| **Semantic Search (Vector Search)** | Uses dense vector embeddings to retrieve content based on underlying conceptual meaning rather than exact word matches. | Natural language queries, conversational questions, synonym/paraphrased matching. |
| **Hybrid Search** | Combines lexical (BM25) and dense vector search, merging score ranks via **Reciprocal Rank Fusion (RRF)**. | <mark style="background-color: #d4edda; color: #155724; padding: 2px 4px; border-radius: 4px;">Industry standard</mark> for general RAG; delivers optimal precision + recall balance. |
| **Metadata Filtering** | Pre-filters or post-filters search spaces based on structured attributes (`author`, `date`, `category`, `tenant_id`). | Scoping queries to specific doc partitions, date ranges, or multi-tenant user access boundaries. |
| **Dense Retrieval** | Retrieves chunks by measuring high-dimensional vector distances (Cosine Similarity, Euclidean L2, Inner Product). | High-accuracy conceptual context matching. |
| **Sparse Retrieval** | Uses term-frequency weighted vectors (**BM25**, **SPLADE**) where most dimensions are zero. | Fast exact word matching with highly interpretable similarity scores. |
| **Multi-Vector Search** | Represents a single document using multiple embeddings (e.g., summary vector + full text vector + image vector). | Complex, multi-topic, or multi-modal documents. |
| **Hierarchical Search** | Performs two-tier retrieval: searches summary/coarse chunks first, then drills down into fine-grained sub-chunks. | Large enterprise manuals, books, lengthy technical specs. |
| **Parent-Child (Recursive) Retrieval** | Matches query against small, highly focused child chunks, but passes larger surrounding parent context to LLM. | Preserving deep context while keeping index chunking fine-grained. |
| **Multi-Query Retrieval** | Uses LLM to generate multiple prompt reformulations of user query, fetching vectors for all variants. | Resolving ambiguous, complex, or underspecified questions. |
| **Self-Query Retrieval** | Uses LLM to parse natural query into structured metadata query + semantic query payload. | Filtered requests (e.g., *"Find finance reports from 2024 covering AI"*). |
| **Graph-Based Retrieval (GraphRAG)** | Navigates entity nodes and relationship edges in a Knowledge Graph alongside vector embeddings. | Highly connected enterprise data, entity tracking, structural relationships. |
| **Reranked Search** | Fetches an expanded candidate pool (`k=20`), then re-scores candidate chunks using Cross-Encoders or Cohere Rerank. | Elevating top-3 context accuracy and removing false positives. |
| **Agentic Retrieval** | Autonomous agent decides dynamically which tool, strategy, or vector collection to query over multiple steps. | Multi-hop reasoning, live database queries, enterprise agent tool execution. |

<br>

#### 🏗️ Common Search Pipeline in Production RAG

```
┌───────────────────────────────────────────────┐
│                  User Query                   │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│     Metadata Filter (Pre-filtering Scope)     │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│                 Hybrid Search                 │
│  ┌────────────────────┬────────────────────┐  │
│  │   BM25 (Sparse)    │  Vector (Dense)    │  │
│  └─────────┬──────────┴─────────┬──────────┘  │
└────────────┼────────────────────┼─────────────┘
             │                    │
             └──────────┬─────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│     Merge Results (Reciprocal Rank Fusion)    │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│      Reranker (Cross-Encoder / Cohere)        │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│        Top-K Context (High Precision)         │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│               LLM Response Output             │
└───────────────────────────────────────────────┘
```

<br>

#### 💡 Core Takeaway for Production Systems:
The most effective and widely adopted stack in enterprise production RAG relies on:
1. **Hybrid Search** (Dense Vector + Sparse BM25) for high retrieval recall.
2. **Metadata Pre-Filtering** to narrow security boundaries and date ranges.
3. **Cross-Encoder Reranking** to ensure top-3 chunks are strictly relevant to the prompt context.

<br>

---

<br>

## 12. Query Enhancement & Advanced RAG (`06_query_enhancment`)

### 12.1 Query Expansion (`1-queryexpansion.ipynb`)
#### Imports
```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
```
#### How to Use
```python
# Expansion prompt template
expansion_prompt = PromptTemplate.from_template("""
You are a helpful assistant. Expand the following query to improve document retrieval by adding relevant synonyms, technical terms, and useful context.

Original query: "{query}"

Expanded query:
""")

expansion_chain = expansion_prompt | llm | StrOutputParser()
expanded_query = expansion_chain.invoke({"query": "What is agent orchestration?"})

# Retrieve documents using expanded query
retrieved_docs = retriever.invoke(expanded_query)
```
#### What They Do
*   `Query Expansion`: Uses an LLM pass to reformulate or enrich short, vague user prompts with technical vocabulary, domain synonyms, and context before retrieval.
*   **Key Concept**: Solves vocabulary mismatch issues where users ask questions using different vocabulary than what is written in the knowledge base.

---

### 12.2 Query Decomposition (`2-querydecomposition.ipynb`)
#### Imports
```python
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableSequence
```
#### How to Use
```python
# 1. Prompt to decompose complex questions into sub-questions
decomposition_prompt = PromptTemplate.from_template("""
Decompose the following complex question into 2 to 4 smaller sub-questions for better document retrieval.

Question: "{question}"

Sub-questions:
""")
decomposition_chain = decomposition_prompt | llm | StrOutputParser()

# 2. Decompose and run sub-retrievals
sub_qs_text = decomposition_chain.invoke({"question": "How does LangChain memory compare to CrewAI?"})
sub_questions = [q.strip("-•1234567890. ").strip() for q in sub_qs_text.split("\n") if q.strip()]

# 3. Retrieve and answer each sub-question independently
results = []
for subq in sub_questions:
    docs = retriever.invoke(subq)
    # Combine retrieved docs for sub-question context
    context = "\n".join(d.page_content for d in docs)
    # Generate sub-answer
    sub_answer = qa_chain.invoke({"question": subq, "context": context})
    results.append(f"Sub-Q: {subq}\nSub-A: {sub_answer}")

# 4. Final Answer Synthesis (Combine sub-answers into comprehensive output)
final_prompt = PromptTemplate.from_template("""Combine the sub-answers to answer the original question: "{question}"\n\nContext:\n{sub_answers}""")
final_chain = final_prompt | llm | StrOutputParser()
final_response = final_chain.invoke({"question": "How does LangChain memory compare to CrewAI?", "sub_answers": "\n\n".join(results)})
```
#### What They Do
*   `Query Decomposition`: Breaks down multi-part or comparative questions into atomic sub-queries that can be retrieved independently.
*   **Key Concept**: Essential for multi-hop RAG pipelines and parallel agentic execution where a single query spans multiple distinct domain areas.

---

### 12.3 Hypothetical Document Embeddings - HyDE (`3-HyDE.ipynb`)
#### Imports
```python
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
```
#### How to Use
```python
# 1. Generate hypothetical document/answer using LLM
hyde_prompt = PromptTemplate.from_template("""
Please write a passage to answer the question.

Question: "{question}"
Passage:
""")
hyde_chain = hyde_prompt | llm | StrOutputParser()
hypothetical_doc = hyde_chain.invoke({"question": "When did Steve Jobs found NeXT?"})

# 2. Embed hypothetical document and search vector store
retrieved_docs = vectorstore.similarity_search(hypothetical_doc, k=3)
```
#### What They Do
*   `HyDE`: Generates a hypothetical response document using an LLM, embeds that hypothetical passage, and uses its vector to search the vector database.
*   **Key Concept**: Eliminates query-document asymmetry. Standard queries are short question sentences, whereas stored chunks are detailed descriptive paragraphs. Vector matching an answer-like text against stored documents yields significantly higher similarity alignment.

---

## 13. Multimodal RAG (`07_multimodle RAG`)

### 13.1 Multimodal RAG with Vision LLMs & Joint Embeddings (`1-multimodalopenai.ipynb`)
#### Imports
```python
import fitz  # PyMuPDF
import io
import base64
from PIL import Image
import torch
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
```
#### How to Use
```python
# 1. Load CLIP model & processor for joint image+text embeddings
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

def embed_image(pil_img):
    inputs = clip_processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        return (features / features.norm(dim=-1, keepdim=True)).squeeze().numpy()

def embed_text(text):
    inputs = clip_processor(text=text, return_tensors="pt", padding=True, max_length=77)
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        return (features / features.norm(dim=-1, keepdim=True)).squeeze().numpy()

# 2. Extract text & images from PDF using PyMuPDF (fitz)
doc = fitz.open("sample.pdf")
all_docs, all_embeddings, image_data_store = [], [], {}
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

for i, page in enumerate(doc):
    # Process text
    text = page.get_text()
    if text.strip():
        for chunk in splitter.split_documents([Document(page_content=text, metadata={"page": i, "type": "text"})]):
            all_embeddings.append(embed_text(chunk.page_content))
            all_docs.append(chunk)

    # Process images
    for img_idx, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        base_img = doc.extract_image(xref)
        pil_img = Image.open(io.BytesIO(base_img["image"])).convert("RGB")
        img_id = f"page_{i}_img_{img_idx}"
        
        # Save base64 for GPT-4V payload
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        image_data_store[img_id] = base64.b64encode(buf.getvalue()).decode()
        
        # Embed image via CLIP
        all_embeddings.append(embed_image(pil_img))
        all_docs.append(Document(page_content=f"[Image: {img_id}]", metadata={"page": i, "type": "image", "image_id": img_id}))

# 3. Create FAISS index with precomputed CLIP embeddings
vector_store = FAISS.from_embeddings(
    text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, np.array(all_embeddings))],
    embedding=None,
    metadatas=[doc.metadata for doc in all_docs]
)

# 4. Multimodal retrieval & Vision LLM invocation (GPT-4V)
llm = init_chat_model("openai:gpt-4.1")
query_emb = embed_text("What does the revenue trend chart show?")
retrieved = vector_store.similarity_search_by_vector(query_emb, k=5)

# Construct message with text + base64 image_url components
message_content = [{"type": "text", "text": "Question: What does the revenue chart show?\n"}]
for doc in retrieved:
    if doc.metadata.get("type") == "text":
        message_content.append({"type": "text", "text": f"Text: {doc.page_content}"})
    elif doc.metadata.get("type") == "image":
        img_id = doc.metadata.get("image_id")
        message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_data_store[img_id]}"}
        })

response = llm.invoke([HumanMessage(content=message_content)])
print(response.content)
```
#### What They Do
*   `CLIP` (`openai/clip-vit-base-patch32`): Multi-modal embedding model that maps both text strings and image pixels into the **same shared vector space**. Allows searching for images using text queries.
*   `PyMuPDF` (`fitz`): High-speed PDF parser extracting embedded raster graphics and raw text.
*   `Base64 Encoding`: Converts image byte streams into inline Base64 data strings for API payload transmission to vision models (GPT-4 Vision / GPT-4o).
*   `Multimodal Prompt Payload`: Constructs structured multi-part message objects containing both string blocks (`{"type": "text"}`) and image URL objects (`{"type": "image_url"}`).
