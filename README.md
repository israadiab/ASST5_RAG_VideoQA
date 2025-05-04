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

## Install dependencies
pip install -r requirements.txt

## Run the Streamlit app
streamlit run app.py

