# RAG (Retrieval-Augmented Generation) Comprehensive Notes - Part 2

---

## 📋 Table of Contents

13. [Multimodal RAG (`07_multimodle RAG`)](#13-multimodal-rag-07_multimodle-rag)
    * [13.1 Multimodal RAG with OpenAI (GPT-4o) & CLIP Joint Embeddings (`1-multimodalopenai.ipynb`)](#131-multimodal-rag-with-openai-gpt-4o--clip-joint-embeddings-1-multimodalopenaiipynb)

---

## 13. Multimodal RAG (`07_multimodle RAG`)

### 13.1 Multimodal RAG with OpenAI (GPT-4o) & CLIP Joint Embeddings (`1-multimodalopenai.ipynb`)

#### 🎯 Why We Need This:
Standard RAG systems rely exclusively on text extraction loaders. However, real-world enterprise documents—such as financial reports, medical charts, technical manuals, architectural designs, and slide decks—contain critical information in visual elements (charts, plots, diagrams, flowcharts, tables, and photos).
- **Text-Only RAG Limitations**: Text splitters skip or mangle visual charts, losing trend data and quantitative relationships.
- **Multimodal RAG Solution**: Combines <mark style="background-color: #d4edda; color: #155724; padding: 2px 4px; border-radius: 4px;">Joint Embedding Space (CLIP)</mark> with <mark style="background-color: #d4edda; color: #155724; padding: 2px 4px; border-radius: 4px;">Vision LLMs (GPT-4o)</mark>.
  - **Shared Vector Space**: Both text passages and image pixels are embedded into the exact same 512-dimensional vector space. A natural language text query (e.g., *"Show Q1 revenue trend chart"*) directly matches image embeddings in a vector database.
  - **Multimodal Prompt Reasoning**: Retrieved text chunks and base64-encoded visual images are assembled into a structured prompt for **GPT-4o**, enabling holistic visual-textual reasoning.

#### 📊 Multimodal AI Architecture Diagram:
```mermaid
flowchart TD
    subgraph Document_Processing["1. Multimodal Document Parsing"]
        Doc["📄 Multimodal Document<br/>(Text + Visual Charts)"]
        TextSplitter["✂️ PyMuPDF & Text Splitter<br/>(Text Chunks)"]
        ImgExtractor["🖼️ Image Extraction & Noise Filter<br/>(PNG -> Base64 URIs)"]
        Doc --> TextSplitter
        Doc --> ImgExtractor
    end

    subgraph Embedding_Space["2. CLIP Joint Vector Space"]
        CLIP_Text["🔤 CLIP Text Encoder"]
        CLIP_Img["👁️ CLIP Vision Encoder (ViT)"]
        L2_Norm["📐 L2 Vector Normalization"]
        VectorDB[("🗄️ Unified Vector Store<br/>(FAISS Index - 512d Space)")]

        TextSplitter --> CLIP_Text
        ImgExtractor --> CLIP_Img
        CLIP_Text --> L2_Norm
        CLIP_Img --> L2_Norm
        L2_Norm --> VectorDB
    end

    subgraph Retrieval_Synthesis["3. Cross-Modal Retrieval & Generation"]
        Query["💬 User Query<br/>(e.g., 'What is the Q1 revenue trend?')"]
        QueryEnc["🔤 Embed Query with CLIP"]
        Search["🔍 Cross-Modal Similarity Search"]
        MsgBuilder["📦 Build Structured Multimodal Message<br/>(Text Context + Base64 Images)"]
        VisionLLM["🧠 Vision LLM (GPT-4o)<br/>(Multimodal Reasoning)"]
        Output["🎯 Final Grounded Answer"]

        Query --> QueryEnc
        QueryEnc --> Search
        VectorDB --> Search
        Search --> MsgBuilder
        ImgExtractor -. "Base64 Data" .-> MsgBuilder
        MsgBuilder --> VisionLLM
        VisionLLM --> Output
    end
```

---

### Imports
```python
import os
import io
import base64
import numpy as np
import torch
from PIL import Image, ImageDraw
import fitz  # PyMuPDF
from dotenv import load_dotenv

# HuggingFace Transformers for CLIP
from transformers import CLIPProcessor, CLIPModel

# LangChain Core & Integration Libraries
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
```

<details>
<summary><b>🧠 Deep Dive: How CLIP Joint Embedding Space & Vector Normalization Work</b></summary>

### How CLIP Enables Cross-Modal Search
OpenAI's CLIP (`openai/clip-vit-base-patch32`) consists of two parallel encoders:
1. **Text Encoder**: Maps text tokens into a 512-dimensional embedding vector \(\mathbf{v}_{text}\).
2. **Image Encoder (Vision Transformer / ViT)**: Maps raw image pixels into a 512-dimensional embedding vector \(\mathbf{v}_{image}\).

Both encoders are contrastively trained on hundreds of millions of image-caption pairs so that text descriptions and corresponding visual images align tightly in the same vector space:

$$\text{Cosine Similarity}(\mathbf{v}_{text}, \mathbf{v}_{image}) = \frac{\mathbf{v}_{text} \cdot \mathbf{v}_{image}}{\|\mathbf{v}_{text}\|_2 \|\mathbf{v}_{image}\|_2}$$

### L2 Vector Normalization Rule
Before storing vectors in FAISS or computing dot products, vectors MUST be L2-normalized to unit magnitude:

$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\sqrt{\sum_{i=1}^{d} v_i^2}}$$

When vectors are normalized (\(\|\hat{\mathbf{v}}\|_2 = 1\)), Inner Product (dot product) equals Cosine Similarity, enabling high-speed vector similarity search.

### Robust HuggingFace `transformers` Output Extraction
Depending on the version of `transformers`, `CLIPModel.get_text_features()` or `get_image_features()` may return a tensor, a `CLIPOutput`, or a `BaseModelOutputWithPooling` object. Robust code extracts the tensor explicitly:

```python
features = clip_model.get_text_features(**inputs)
if hasattr(features, "pooler_output"):
    features = features.pooler_output
elif hasattr(features, "text_embeds"):
    features = features.text_embeds
elif isinstance(features, (tuple, list)):
    features = features[0]

# Normalize to unit vector
features = features / features.norm(dim=-1, keepdim=True)
```
</details>

<details>
<summary><b>💡 Helper Pattern: Programmatically Generating Synthetic Multimodal Sample PDFs</b></summary>

When developing and testing multimodal RAG pipelines without requiring external sample downloads, you can programmatically generate a synthetic PDF containing both text and visual image charts using PyMuPDF (`fitz`) and `Pillow`:

```python
def ensure_sample_pdf(pdf_path="multimodal_sample.pdf"):
    if os.path.exists(pdf_path):
        return pdf_path
    
    doc = fitz.open()
    page = doc.new_page()
    
    # 1. Add text elements
    page.insert_text((50, 50), "Sample Multimodal PDF Document", fontsize=18)
    page.insert_text((50, 80), "This document contains text descriptions and visual chart images.", fontsize=12)
    page.insert_text((50, 110), "Company Revenue for Q1 reached $10M with strong growth in cloud services.", fontsize=11)
    
    # 2. Draw visual chart image using PIL
    img = Image.new('RGB', (300, 150), color=(240, 240, 240))
    d = ImageDraw.Draw(img)
    d.rectangle([30, 30, 270, 120], outline=(0, 0, 255), width=2)
    d.text((40, 40), "Q1 Revenue Chart: $10M", fill=(0, 0, 0))
    d.line([(50, 100), (100, 80), (150, 90), (200, 50), (250, 40)], fill=(255, 0, 0), width=3)
    
    # 3. Stream image bytes to PDF page
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    page.insert_image(fitz.Rect(50, 150, 350, 300), stream=img_bytes.getvalue())
    
    doc.save(pdf_path)
    doc.close()
    return pdf_path
```
*   **Why use this?** Guarantees that unit tests and demonstration notebooks remain 100% self-contained and reproducible across any environment.
</details>

<details>
<summary><b>🖼️ OpenAI Vision API Multimodal Message Payload & Detail Level Rules</b></summary>

OpenAI's Vision API (used by `gpt-4o` and `gpt-4o-mini`) accepts multi-part structured `HumanMessage` content payloads combining string blocks and Data URI image blocks:

```python
HumanMessage(content=[
    {
        "type": "text", 
        "text": "Question: What does the chart show about revenue trends?"
    },
    {
        "type": "text", 
        "text": "Context Excerpts:\n[Page 0]: Revenue reached $10M in Q1."
    },
    {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
            "detail": "auto"  # Options: 'auto', 'low', 'high'
        }
    }
])
```

### Vision Detail Parameter Settings
- `detail: "low"`: Converts image to a fixed 512x512 low-res tile (<mark style="background-color: #d4edda; color: #155724; padding: 2px 4px; border-radius: 4px;">85 tokens per image</mark>). Fast & cheap for simple layouts or logos.
- `detail: "high"`: Crops & processes high-resolution 512px tiles (<mark style="background-color: #fff3cd; color: #856404; padding: 2px 4px; border-radius: 4px;">170 tokens per tile + 85 base tokens</mark>). Best for reading small diagram labels and intricate chart numbers.
- `detail: "auto"`: Model dynamically selects low or high detail based on image resolution (<mark style="background-color: #d4edda; color: #155724; padding: 2px 4px; border-radius: 4px;">Recommended default</mark>).
</details>

---

### How to Use

#### 1. Initializing CLIP & Custom Embeddings Wrapper
```python
# Load CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

class CLIPEmbeddings(Embeddings):
    """Custom Embeddings wrapper implementing LangChain's Embeddings interface for CLIP."""
    def __init__(self, model, processor, device=device):
        self.model = model
        self.processor = processor
        self.device = device

    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            elif hasattr(features, "text_embeds"):
                features = features.text_embeds
            elif isinstance(features, (tuple, list)):
                features = features[0]
            features = features / features.norm(dim=-1, keepdim=True)
            return features.squeeze(0).cpu().numpy().astype(np.float32)

    def embed_image(self, image_input) -> np.ndarray:
        image = Image.open(image_input).convert("RGB") if isinstance(image_input, str) else image_input.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            elif hasattr(features, "image_embeds"):
                features = features.image_embeds
            elif isinstance(features, (tuple, list)):
                features = features[0]
            features = features / features.norm(dim=-1, keepdim=True)
            return features.squeeze(0).cpu().numpy().astype(np.float32)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t).tolist() for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text).tolist()

clip_embedder = CLIPEmbeddings(clip_model, clip_processor)
```

#### 2. Parsing PDF Text & Extracting Filtered Images (`PyMuPDF`)
```python
# Open PDF document
doc = fitz.open("multimodal_sample.pdf")

all_docs = []
all_embeddings = []
image_data_store = {}

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

for i, page in enumerate(doc):
    # 1. Process text content
    text = page.get_text()
    if text.strip():
        temp_doc = Document(page_content=text, metadata={"page": i, "type": "text"})
        chunks = splitter.split_documents([temp_doc])
        for chunk in chunks:
            all_embeddings.append(clip_embedder.embed_text(chunk.page_content))
            all_docs.append(chunk)

    # 2. Extract visual image elements
    for img_idx, img in enumerate(page.get_images(full=True)):
        try:
            xref = img[0]
            base_img = doc.extract_image(xref)
            pil_img = Image.open(io.BytesIO(base_img["image"])).convert("RGB")
            
            # 🔹 RULE OF THUMB: Filter graphic noise (icons, bullet dots, lines)
            if pil_img.width < 50 or pil_img.height < 50:
                continue
                
            img_id = f"page_{i}_img_{img_idx}"
            
            # Store base64 data URI in memory for Vision LLM
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            image_data_store[img_id] = base64.b64encode(buf.getvalue()).decode()
            
            # Compute CLIP embedding for image
            all_embeddings.append(clip_embedder.embed_image(pil_img))
            all_docs.append(Document(
                page_content=f"[Image: {img_id}]",
                metadata={"page": i, "type": "image", "image_id": img_id}
            ))
        except Exception as e:
            print(f"Error processing image {img_idx} on page {i}: {e}")

doc.close()
```

#### 3. Building Unified Vector Store (`FAISS`)
```python
# Convert embeddings to numpy array
embeddings_array = np.array(all_embeddings, dtype=np.float32)
text_embedding_pairs = [(doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)]

# Build FAISS vector store with precomputed CLIP embeddings
vector_store = FAISS.from_embeddings(
    text_embeddings=text_embedding_pairs,
    embedding=clip_embedder,
    metadatas=[doc.metadata for doc in all_docs]
)
```

#### 4. Multimodal Retrieval & Vision LLM Synthesis (`GPT-4o`)
```python
# Initialize ChatOpenAI model (with dry-run key fallback)
openai_key = os.getenv("OPENAI_API_KEY", "sk-placeholder-key-for-dry-run")
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_key)

def retrieve_multimodal(query: str, k: int = 5):
    """Search unified vector space using text query CLIP embedding."""
    query_emb = clip_embedder.embed_text(query)
    return vector_store.similarity_search_by_vector(embedding=query_emb, k=k)

def create_multimodal_message(query: str, retrieved_docs: list[Document]) -> HumanMessage:
    """Construct structured HumanMessage with text excerpts and inline base64 images."""
    content = [{"type": "text", "text": f"Question: {query}\n\nContext:\n"}]
    
    text_docs = [d for d in retrieved_docs if d.metadata.get("type") == "text"]
    image_docs = [d for d in retrieved_docs if d.metadata.get("type") == "image"]
    
    if text_docs:
        context_str = "\n\n".join([f"[Page {d.metadata['page']}]: {d.page_content}" for d in text_docs])
        content.append({"type": "text", "text": f"Text Excerpts:\n{context_str}\n"})
        
    for d in image_docs:
        img_id = d.metadata.get("image_id")
        if img_id and img_id in image_data_store:
            content.append({"type": "text", "text": f"\n[Image from page {d.metadata['page']}]:\n"})
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data_store[img_id]}",
                    "detail": "auto"
                }
            })
            
    content.append({"type": "text", "text": "\n\nPlease answer the question based on the provided text and images."})
    return HumanMessage(content=content)

def multimodal_pdf_rag_pipeline(query: str):
    retrieved = retrieve_multimodal(query, k=5)
    message = create_multimodal_message(query, retrieved)
    
    real_key = os.getenv("OPENAI_API_KEY")
    if not real_key or real_key.startswith("sk-placeholder"):
        print("Note: OPENAI_API_KEY is missing. Skipping LLM invocation.")
        return "[Dry-run complete: OpenAI API key required for full answer generation]"
        
    response = llm.invoke([message])
    return response.content

# Execute query
response_text = multimodal_pdf_rag_pipeline("What does the revenue trend chart show?")
print(response_text)
```

---

### What They Do
*   `CLIPModel` & `CLIPProcessor` (`openai/clip-vit-base-patch32`): Multi-modal neural network that projects text strings and image pixels into the **same shared 512-dimensional vector space**, enabling cross-modal text-to-image similarity search.
*   `CLIPEmbeddings`: Custom subclass of `langchain_core.embeddings.Embeddings`. Wraps CLIP text and image encoding methods into LangChain standard `embed_documents` and `embed_query` contracts.
*   `PyMuPDF` (`fitz`): C-accelerated PDF parser. Performs rapid page text extraction and binary image stream extraction (`doc.extract_image`).
*   `Image Dimension Filter`: <mark style="background-color: #d4edda; color: #155724; padding: 2px 4px; border-radius: 4px;">Crucial pre-processing step</mark>. Discards images with `width < 50` or `height < 50` pixels to eliminate bullet points, lines, background fills, and UI icons from polluting vector search.
*   `FAISS.from_embeddings`: Vector database index initialized with pre-computed text and image embedding tuples `(content, embedding_vector)` alongside custom metadata.
*   `ChatOpenAI(model="gpt-4o")`: OpenAI's flagship multimodal model capable of reading and reasoning over both text strings and high-resolution images.
*   `HumanMessage(content=[...])`: Structured multi-part message object carrying text prompts (`"type": "text"`) and inline base64 image objects (`"type": "image_url"`).

---

### 💡 Advanced Best Practices & Key Insights:
*   **Cross-Modal Retrieval Efficiency**: Because CLIP places text and image vectors in the exact same space, a user query like *"revenue graph"* directly retrieves the corresponding image document vector even if no OCR text was extracted from that image.
*   **Image Format & RGB Conversion**: Always convert extracted PDF image bytes using `.convert("RGB")`. PDFs often store images in CMYK, RGBA, or indexed palette formats that crash PyTorch or OpenAI vision models if not converted to standard RGB PNG/JPEG.
*   **Base64 Memory Management**: Store base64 data strings in an in-memory key-value dictionary (`image_data_store`) or cloud object storage (S3/GCS) indexed by `image_id` (`page_X_img_Y`). Avoid dumping thousands of temporary image files onto local disk.
*   **Token Cost & Detail Level Optimization**: Always set `"detail": "auto"` (or `"low"` for high-volume pipelines) on `image_url` dictionary objects to prevent excessive vision token consumption on simple diagrams.
