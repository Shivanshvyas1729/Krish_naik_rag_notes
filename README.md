# Krish_naik_rag_notes
<details><summary>Rag</summary>
Here are structured study notes based on the provided document about RAG:

# Study Notes: Retrieval-Augmented Generation (RAG)

## Core Concept

**RAG (Retrieval-Augmented Generation)** is a technique that enhances AI language models by combining their text-generation capabilities with external knowledge retrieval.

* **The Analogy:**
* **Traditional LLM (Without RAG):** Like a student taking a *closed-book exam*. It can only answer based on the information it memorized during its initial training. If it doesn't know, it might guess (hallucinate) or say "I don't know."
* **RAG-Enabled AI:** Like a student taking an *open-book exam*. It can look up specific, current, or specialized information from a "library" (external databases) before generating its answer.



---

## The 3 Core Components of RAG

1. **[R]etrieval:** Finding relevant information. The system searches external sources (like a Vector Database) using similarity search to find data related to the user's query.
2. **[A]ugmentation:** Enhancing the context. The retrieved data is combined with metadata (e.g., source tags like *"Source: Tesla Annual Report 2023"*) and added to the user's original prompt.
3. **[G]eneration:** Producing the answer. The Large Language Model (LLM) reads the enriched context and generates a highly accurate, grounded response.

---

## RAG Architecture Workflow

The process is broken down into three distinct phases:

### Phase 1: Document Ingestion

* **Data Sources:** Raw data (PDFs, Web Pages, Databases) is collected.
* **Processing:** The data goes through a Document Splitter to break it into chunks.
* **Embedding:** An Embedding Model converts text into mathematical vectors (e.g., `[0.31, -0.22, 0.85...]`).
* **Storage:** These vectors are stored in a **Vector Database**.

### Phase 2: Query Processing

* The user submits a query (e.g., "What is RAG?").
* The query is converted into an embedding.
* The system performs a **Similarity Search** in the Vector Database to find the most relevant document chunks.

### Phase 3: Generation

* The relevant chunks are formatted as **Augmented Context**.
* This context is fed into a **Large Language Model** (like GPT-4, Claude, or Llama).
* The LLM synthesizes the information and outputs the **Generated Response**.

---

## Comparison: Traditional LLM vs. RAG (Customer Support Example)

| Feature | Traditional LLM (Without RAG) | AI Assistant (With RAG) |
| --- | --- | --- |
| **Data Source** | Training data only | LLM + Vector Database |
| **Response Type** | Generic, unhelpful, or outdated | Specific, actionable, and up-to-date |
| **Example Output** | *"Generally, most companies offer 30-day returns, but policies may vary..."* | *"According to our current policy (v3.2), Black Friday purchases have an extended 60-day window..."* |

---

## Real-World Benefits & Business Impact

* **Cost Savings:** Reduces the need for constant model retraining. *(Example: JPMorgan saved $150M annually by using RAG instead of fine-tuning models monthly).*
* **Accuracy (Reducing Hallucinations):** Grounds the AI in actual facts. *(Example: Microsoft reported a 94% reduction in AI hallucinations in their Copilot products).*
* **Flexibility & Real-Time Updates:** Can ingest live data instantly. *(Example: Bloomberg updates its financial AI assistant hourly with new market data, which is impossible with traditional LLMs).*
* **Compliance & Sourcing:** Allows the AI to provide citations. *(Example: Healthcare companies use RAG to ensure AI responses always cite approved medical sources).*

  [1-2RAG (1).pdf](https://github.com/user-attachments/files/29892069/1-2RAG.1.pdf)

 </details>




<details><summary>fine tunning vs rag </summary>
Here are your structured notes based on the document. Since I cannot extract and embed the images directly from your local PDF, I have recreated the visual flow of each diagram using text-based flowcharts to ensure you have the complete picture.

---

# AI Customization Methods: A Beginner's Guide

A comparison of the three primary ways to customize Large Language Models (LLMs): Prompt Engineering, Fine-tuning, and RAG.

## 1. Prompt Engineering

**Concept:** Teaching through instructions. The underlying AI model itself remains completely unchanged.

### 📊 Diagram Flow

```text
[User Prompt: "Act as an expert chef..."] 
                    ↓
        [Base LLM (Remains Unchanged)] 
                    ↓
          [Customized Output]

```

### 📝 Key Details

* **How it Works:**
* Write specific instructions in your prompt.
* Structure prompts with clear context.
* Use examples (few-shot learning).


* **Pros:**
* No technical expertise needed.
* Instant results.
* Free (no training costs).
* Highly flexible and works with any LLM.


* **Cons:**
* Strictly limited by the model's existing base knowledge.
* Can yield inconsistent results.
* Token limits restrict how complex you can make the prompt.
* Cannot add new, permanent knowledge to the model.


* **Best For:** Quick prototyping, small-scale applications, general-purpose tasks, and when you need maximum flexibility.

---

## 2. Fine-Tuning

**Concept:** Teaching through training. It alters the model's permanent weights to create a specialized version of the original AI.

### 📊 Diagram Flow

```text
[Base LLM (Original Weights)]  +  [Domain-Specific Training Data]
                               ↓
                            (Train)
                               ↓
        [Fine-Tuned LLM (Modified Weights / Specialized)]

```

### 📝 Key Details

* **How it Works:**
* Prepare domain-specific training data.
* Train the base model on your data.
* Model weights are permanently changed to create a specialized version.


* **Pros:**
* Creates deeply specialized knowledge and consistent behavior.
* Eliminates the need for complex prompt engineering.
* Can learn specific new writing styles.
* Significantly better for highly specific domains.


* **Cons:**
* Expensive to execute (can cost $1000s – $10000s).
* Requires Machine Learning (ML) expertise.
* Needs complete retraining for any informational updates.
* The model can sometimes "forget" general knowledge during training.


* **Best For:** Highly specific writing styles or tones, domain-specific language, high-volume/consistent tasks, and situations where accuracy is critical.

---

## 3. RAG (Retrieval-Augmented Generation)

**Concept:** Teaching through retrieval. It pulls in outside information in real-time to help the AI answer a query accurately.

### 📊 Diagram Flow

```text
[User Query] ─────────────> [Vector Database / Knowledge Base]
      ↓                                   ↓
      └─────────> [Retrieved Relevant Documents]
                                  ↓
                              [Base LLM]
                                  ↓
                        [Augmented Response]

```

### 📝 Key Details

* **How it Works:**
* Store company documents or data in a Vector Database.
* Retrieve relevant documents for each specific query.
* Combine the retrieved documents with the query to serve as context.
* The LLM generates an answer based strictly on that context.


* **Pros:**
* Always provides up-to-date information.
* Requires no model training (highly cost-effective).
* Can safely handle private or proprietary data.
* High accuracy with reduced hallucination.


* **Cons:**
* Requires initial infrastructure setup (like Vector DBs).
* The final result is heavily dependent on the quality of the retrieval step.
* Context window limitations still apply.
* Adds latency (delay) to the response time due to the retrieval step.


* **Best For:** Knowledge bases and documentation, real-time or frequently updated info, customer support systems, and compliance-heavy industries.

  [5-Promptvsfinetunignvsrag.pdf](https://github.com/user-attachments/files/29892064/5-Promptvsfinetunignvsrag.pdf)

  </details>




<details><summary>Vector store / vector databases</summary>Here are proper, structured notes based on the document:

# Study Notes: Vector Stores vs. Vector Databases

## The Golden Rule

Start with a **Vector Store** for prototyping and learning. Graduate to a **Vector Database** when you need production-scale features, reliability, and advanced querying capabilities.

---

## 1. Vector Stores

A lightweight library or tool focused on storing and searching vectors efficiently.

### Key Characteristics

* **Core Function:** Simple similarity search (finding the K nearest neighbors to a query vector).
* **Architecture:** Usually runs in-memory or as a local file (single-machine operation).
* **Scale:** Handles smaller datasets (< 1 million vectors).
* **Speed:** Extremely fast query speed (Microseconds).
* **Setup & Cost:** Quick to set up (Minutes), typically deployed locally, and usually free.

### When to Use

* Building a proof of concept (POC).
* Working with less than 1 million vectors.
* You need the absolute fastest possible search speed.
* You have a limited budget or want full control over the implementation.
* Building embedded applications.

### Popular Examples

* FAISS
* Annoy
* ChromaDB
* ScaNN
* NMSLIB

---

## 2. Vector Databases

A full-featured database system designed for managing and querying vector data at scale.

### Key Characteristics

* **Core Function:** Advanced search (filters, metadata queries) and full database operations (CRUD: Create, Read, Update, Delete).
* **Architecture:** Distributed system with replication, sharding, and high availability.
* **Scale:** Built for massive datasets (Billions+ of vectors).
* **Speed:** Slightly slower query speed due to overhead (Milliseconds).
* **Setup & Cost:** Takes longer to set up (Hours/Days), usually cloud-deployed, and incurs costs ($$$).

### When to Use

* Building production and enterprise applications.
* Need to scale beyond millions of vectors.
* Require high availability and system reliability.
* Need advanced filtering and metadata search alongside vector search.
* Have multiple users/tenants accessing the data.
* Want managed infrastructure rather than handling it locally.

### Popular Examples

* Pinecone
* Weaviate
* Qdrant
* Milvus
* Vespa
* DataStax

---

## 3. Quick Reference Comparison

| Feature | Vector Store | Vector Database |
| --- | --- | --- |
| **Scale** | ~1 Million vectors | Billions+ vectors |
| **Setup Time** | Minutes | Hours/Days |
| **Query Speed** | Microseconds | Milliseconds |
| **Features** | Basic Search | Full CRUD & Metadata filtering |
| **Deployment** | Local | Cloud |
| **Cost** | Free | Paid ($$$) |
[23-+Vector+store+vs+Vector+Databases.pdf](https://github.com/user-attachments/files/29892056/23-%2BVector%2Bstore%2Bvs%2BVector%2BDatabases.pdf)

</details>



Semantic Chunking is a text-splitting technique that divides content based on meaning instead of fixed size or paragraphs.
<details><summary>sementic chunking </summary>Here is a summary of the provided document on **Semantic Chunking**:

## Overview

**Semantic Chunking** is the process of splitting a document into meaningful units (chunks) based on **semantic similarity** rather than fixed criteria like token count or line numbers.

In Retrieval-Augmented Generation (RAG) systems, semantic chunking improves performance through the pipeline:

$$\text{Better chunks} \rightarrow \text{Better retrieval} \rightarrow \text{Better grounding} \rightarrow \text{Better answers}$$

Chunks generated via this method are designed to be **self-contained, contextually rich, and logically separated**.

---

## How It Works (Step-by-Step)

1. **Document Segmentation:** The document is split into smaller units, such as individual sentences or paragraphs.
2. **Sentence Embedding:** Each sentence/unit is converted into a vector representation using an embedding model.
3. **Semantic Similarity Check:** The similarity (e.g., Cosine Similarity) between adjacent sentence embeddings is calculated and compared against a defined threshold (e.g., $0.80$).
4. **Sentence Merging:** Adjacent sentences are merged into a single chunk if their similarity score meets or exceeds the threshold.
5. **Form Chunks:** The process outputs grouped chunks containing semantically related sentences, while distinct sentences are separated into standalone chunks.

---

## Example

Given the input text:

1. *"LangChain is a framework for building LLM-powered apps."*
2. *"It integrates with tools like OpenAI and Pinecone."*
3. *"The Eiffel Tower is located in Paris."*
4. *"France is a popular tourist destination."*

**Output Chunks:**

* **Chunk 1:** `["LangChain is a framework...", "It integrates with tools..."]` *(Merged because both discuss LangChain/LLMs)*
* **Chunk 2:** `["The Eiffel Tower is located in Paris."]`
* **Chunk 3:** `["France is a popular tourist destination."]`

  [33-Semantic+Chunking.pdf](https://github.com/user-attachments/files/29892074/33-Semantic%2BChunking.pdf)
</details>



<details><summary>Dense+Sparse retrival</summary>
Here are the structured notes based on the document you shared:

## Hybrid Search Strategies: Dense & Sparse Retrieval

**Hybrid Retrieval** combines both dense and sparse scoring methods (e.g., using a weighted sum or learning-to-rank methods) to improve search recall and relevance. By combining these, you get the "best of both worlds": the semantic, context-aware power of vector embeddings and the precise, exact-match capabilities of keywords.

---

### 1. Sparse Retrieval (Exact Keyword Search)

Sparse retrieval focuses on finding exact word matches between the query and the documents.

* **How it works:** It converts text into a sparse matrix representing word occurrences.
* **Techniques used:** Bag-of-Words (BoW), TF-IDF, and BM25.
* **Best for:** Exact keyword searches (e.g., searching for a specific name, ID, or unique term).

### 2. Dense Retrieval (Semantic Search)

Dense retrieval focuses on the underlying *meaning* and context of the query rather than just exact word matches.

* **How it works:** It uses Vector Embeddings to map text into a high-dimensional vector space. It finds matches by calculating the similarity between the query vector and document vectors.
* **Techniques used:** Cosine Similarity.
* **Common Tools:** Vector databases like FAISS and ChromaDB.
* **Best for:** Semantic meaning (e.g., knowing that "building apps" and "developing software" mean similar things).

---

### The Hybrid Search Formula

Hybrid search calculates a final score by combining the dense and sparse scores using a specific weightage ($\alpha$).

**The Equation:**


$$\text{Score}_{\text{hybrid}} = \alpha \times \text{Score}_{\text{dense}} + (1 - \alpha) \times \text{Score}_{\text{sparse}}$$

**Where:**

* $\text{Score}_{\text{dense}}$ is calculated using Cosine Similarity between the input and the vector store.
* $\text{Score}_{\text{sparse}}$ is calculated using techniques like TF-IDF.
* $\alpha$ is the weightage (often set to $0.5$ for an equal balance).

---

### Practical Example

**Documents in Database:**

* **D1:** "LangChain helps build LLM apps"
* **D2:** "Pinecone is used for vector search"
* **D3:** "Eiffel Tower is in Paris"

**User Query:** "build application using LLM"
**Weightage ($\alpha$):** 0.5

*(Assuming hypothetical individual scores for demonstration)*

* **D1 Calculation:**
* Dense Score = 0.85, Sparse Score = 0.60
* $\text{D1 Score} = (0.5 \times 0.85) + (0.5 \times 0.60) = 0.725$


* **D2 Calculation:**
* Dense Score = 0.40, Sparse Score = 0.20
* $\text{D2 Score} = (0.5 \times 0.40) + (0.5 \times 0.20) = 0.30$


* **D3 Calculation:**
* Dense Score = 0.10, Sparse Score = 0.10
* $\text{D3 Score} = (0.5 \times 0.10) + (0.5 \times 0.10) = 0.10$



**Result:** D1 has the highest hybrid score, making it the most relevant document returned for the query.
</details>


You can find the documentation in the [Text Representation tech. Repo](https://github.com/Shivanshvyas1729/pydantic_notes/blob/main/nlp/Text%20Representation%20tech.md).

<details><summary>Reranking </summary>
## Study Notes: Hybrid Search Strategies & Re-Ranking Techniques
<img width="537" height="641" alt="image" src="https://github.com/user-attachments/assets/68528d95-1e6b-41b5-86c4-ad2136e86cb0" />


### 1. Overview of Re-Ranking

* **Definition:** Re-ranking is a **second-stage filtering process** used in retrieval systems, particularly within Retrieval-Augmented Generation (RAG) pipelines.
* **Core Objective:** To refine and re-order an initial set of retrieved document chunks so that the most relevant contextual evidence appears at the top before being sent to the LLM.

---

### 2. RAG Pipeline Stages & Architecture

The workflow is divided into three distinct stages:

1. **Stage 1: Retrieval (Fast, Broad Retrieval)**
* **Exact Match Retrieval:** Uses algorithms like **BM25** to find literal keyword matches.
* **Semantic Search:** Uses vector store embeddings (e.g., **FAISS**) to match documents by semantic similarity.
* **Hybrid Search:** Combines results from both exact keyword search and vector similarity search to produce an initial `top-k` set of candidate chunks.


2. **Stage 2: Re-Ranking (Accurate, Deep Re-Scoring)**
* Takes the `top-k` candidates from Stage 1.
* Uses a slower but significantly more accurate model—such as a **Cross-Encoder** or an **LLM**—to evaluate the full query-document pair.
* Re-scores and reorders the chunks to select only the highest-quality relevant context.


3. **Stage 3: Generation**
* Feeds the user prompt alongside the top re-ranked relevant chunks into the LLM to generate the final response.



---

### 3. Why Use Re-Rankers in a RAG Pipeline?

| Factor / Strategy | Without Re-Ranker | With Re-Ranker |
| --- | --- | --- |
| **1. Relevance of Context** | `Top-k` documents may only be loosely or partially related. | `Top-k` documents are re-scored and reordered for maximum relevance. |
| **2. Factual Accuracy** | LLMs are prone to hallucinations if low-quality context is retrieved. | Irrelevant documents are filtered out, resulting in grounded, factual answers. |
| **3. Handling Ambiguity** | First-stage retrievers lack a deep understanding of complex query intent. | Evaluates full query-doc pairs for significantly better intent alignment. |
| **4. Semantic Matching** | Dense retrievers can miss relevant documents that have low vector similarity scores. | Leverages deeper models (cross-encoders/LLMs) to capture subtle semantic connections. |
| **5. Keyword vs. Meaning** | Keyword models (like BM25) may favor exact string matches even if contextually unhelpful. | Effectively balances lexical (keyword) and semantic (meaning) relevance. |
| **6. Evidence Prioritization** | All retrieved documents are treated with equal importance. | The highest-quality evidence is dynamically floated to the top. |
| **7. Long-Tail Queries** | Weak retrievers struggle to locate matches for rare or niche queries. | Better captures rare, long-tail, but highly meaningful matches. |
| **8. LLM Efficiency** | Irrelevant context causes LLMs to yield verbose, unfocused, or incorrect output. | High-precision context improves generation speed, conciseness, and accuracy. |
| **9. Noise Reduction** | Unrelated content (e.g., ads, boilerplate text) can slip into the prompt. | Pushes noisy content to the bottom or filters it out entirely. |
| **10. Flexible Scoring** | Constrained to fixed retriever scoring rules. | Allows custom scoring strategies incorporating metadata, recency, or user preferences. |

---

### 4. Summary Takeaway

> **First-stage retrievers** prioritize **speed** over precision to fetch candidate chunks from large databases. **Second-stage re-rankers** trade speed for **accuracy** by evaluating candidate chunks through a deeper neural network, ensuring the LLM context window receives only clean, prioritized, and highly factual information.


</details>

<details><summary>MMR (Maximal Marginal Relevance )</summary>

Here are your organized notes based on the document provided:

# Hybrid Search Strategies: Maximal Marginal Relevance (MMR)

## What is MMR?

**Maximal Marginal Relevance (MMR)** is a powerful, diversity-aware retrieval technique used primarily in information retrieval and Retrieval-Augmented Generation (RAG) pipelines.

**Aim:** Its primary goal is to balance **relevance** and **novelty**. It prevents the retriever from returning highly similar documents that repeat the same content. MMR ensures selected documents are both:

1. Relevant to the user's query.
2. Diverse from one another (non-redundant).

---

## The MMR Formula

The algorithm evaluates a candidate document's relevance against the query while penalizing it for similarity to documents that have already been selected.

$$\text{MMR}(d) = \lambda \cdot \text{sim}(d, q) - (1 - \lambda) \cdot \max_{s \in S} \text{sim}(d, s)$$

### Key Parameters:

* $q$: The user query.
* $d$: A candidate document from the document set $D$.
* $S$: The set of documents that have *already* been selected.
* $\text{sim}(a, b)$: The similarity function being used (e.g., Cosine Similarity).
* $\lambda$ (Lambda): A tunable parameter between $0$ and $1$.
* A higher $\lambda$ prioritizes **relevance** to the query.
* A lower $\lambda$ prioritizes **diversity** among documents.



---

## Step-by-Step Example

Imagine we have three candidate documents (D1, D2, D3) and we want to select the top 2 documents using MMR.

**1. Initial Query Relevance (Cosine Similarity)**

* $\text{sim}(D1, Q) = 0.95$
* $\text{sim}(D2, Q) = 0.93$
* $\text{sim}(D3, Q) = 0.80$

**Step 1:** We pick **D1** first because it has the highest raw similarity score (0.95).

**2. Calculating Diversity (Similarity to Selected Doc D1)**

* $\text{sim}(D1, D2) = 0.90$ (Highly redundant)
* $\text{sim}(D1, D3) = 0.30$ (Highly diverse)

**Step 2:** Select the second document using the MMR formula. Let's assume $\lambda = 0.7$.

* **For D2:**

$$\text{MMR}(D2) = (0.7 \cdot 0.93) - (0.3 \cdot 0.90) = 0.651 - 0.270 = \mathbf{0.381}$$


* **For D3:**

$$\text{MMR}(D3) = (0.7 \cdot 0.80) - (0.3 \cdot 0.30) = 0.560 - 0.090 = \mathbf{0.470}$$



**Result:** Even though D2 is more relevant to the query than D3 ($0.93$ vs $0.80$), **D3** is selected as the second document.

* **Final Rank:** 1. D1 | 2. D3
* **Reason:** D3 provides the best balance of diversity and relevance, whereas D2 was too redundant with the information already present in D1.

---

## When to Use vs. When Not to Use MMR

| Scenario | Details / Reasoning |
| --- | --- |
| **When to Use MMR** |  |
| **RAG Pipelines** | Avoids feeding Large Language Models (LLMs) redundant documents, leading to richer, more useful context. |
| **Chatbots & Search Apps** | Great for FAQs, document browsers, and applications where a broad coverage of an answer is needed. |
| **Hybrid Retrieval** | Works well when combining Dense + Sparse search strategies. |
| **When NOT to Use MMR** |  |
| **Extremely Short Context** | If you only have room for (or only want) the single top-1 most relevant document. |
| **Precision Only** | When you are strictly focused on accuracy and do not care about topic coverage. |
| **Pre-existing Diversity** | If the source documents are already inherently diverse. |
| **LLM Reranking** | If redundancy is already being handled downstream by an LLM post-filter or reranker. |
</details>
