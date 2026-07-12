# Krish_naik_rag_notes



[1-2RAG (1).pdf](https://github.com/user-attachments/files/29892069/1-2RAG.1.pdf)

[5-Promptvsfinetunignvsrag.pdf](https://github.com/user-attachments/files/29892064/5-Promptvsfinetunignvsrag.pdf)

[23-+Vector+store+vs+Vector+Databases.pdf](https://github.com/user-attachments/files/29892056/23-%2BVector%2Bstore%2Bvs%2BVector%2BDatabases.pdf)

[33-Semantic+Chunking.pdf](https://github.com/user-attachments/files/29892074/33-Semantic%2BChunking.pdf)


Semantic Chunking is a text-splitting technique that divides content based on meaning instead of fixed size or paragraphs.

Process (short):

Convert the text into sentences.
Generate an embedding (semantic vector) for each sentence.
Measure similarity between consecutive sentences.
Group highly similar sentences into the same chunk.
Start a new chunk when similarity drops below a chosen threshold.
Optionally enforce minimum/maximum chunk sizes.

Result: Chunks contain semantically related information, improving retrieval quality in RAG systems compared to fixed-size chunking.
