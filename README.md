# Retrieval-Augmented Generation (RAG) for Video Question Answering

This project implements a **Video Question Answering (Video QA)** system that supports multiple retrieval techniques to answer user queries based on [this video](https://www.youtube.com/watch?v=dARr3lGKwk8). 

***Retrieval methods supported***:
- **FAISS**: Fast semantic search using vector similarity (Semantic Search)
- **TF-IDF**: Traditional keyword-based search (Lexical Search)
- **BM25**: Advanced lexical search method used widely in information retrieval (Lexical Search)
- **PostgreSQL (pgvector)**: Semantic search powered by pgvector indexes (HNSW and IVFFLAT) hosted on Supabase. (Semantic Search)

---

## Project Structure
- app.py # Streamlit app
- data_ingestion_pipeline.ipynb # Preprocessing and data preparation notebook
- faiss.index # FAISS index for semantic retrieval
- full_transcript.txt # Raw video transcript
- image_embeddings.npy # For future work (currently unused)
- text_embeddings.npy # Sentence embeddings for transcript chunks
- transcript.json # Processed transcript chunks
- upload_embeddings_to_supabase.py # Script to upload embeddings to Supabase
- frames/ # Video frames (if used later)

## Run the Streamlit App
1. Clone the repository
2. Install requirements
   ```bash
   pip install -r requirements.txt
4. Run the streamlit app
   ```bash
   streamlit run app.py

## PostgreSQL / Supabase Requirement
- To use the PostgreSQL pgvector retrieval option, you must have a [Supabase]([url](https://supabase.com/)) project set up.
- The transcript_chunks table must be populated with the transcript chunks and their embeddings.
- Indexes (ivfflat and hnsw) should be created on the embedding column for efficient semantic search.
- If you don’t have access to the Supabase database, the FAISS, TF-IDF, and BM25 retrieval methods will still work locally.


