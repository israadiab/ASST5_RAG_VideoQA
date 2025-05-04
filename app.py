import streamlit as st
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import time
from supabase import create_client, Client

# ---------- SETUP ----------

st.set_page_config(page_title="RAG Video QA App", layout="wide")

# Supabase credentials
url = "https://fargcblkexcrmfiztvkz.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZhcmdjYmxrZXhjcm1maXp0dmt6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDYzNTkwNTAsImV4cCI6MjA2MTkzNTA1MH0.ksSgFWTY-F8O_qIGYLTFSx7EQm74bIc6c7nsr43Nd2k"
supabase: Client = create_client(url, key)

# ---------- DATA LOADING ----------

@st.cache_resource
def load_resources():
    with open("transcript.json", "r") as f:
        chunks = json.load(f)

    text_embeddings = np.load("text_embeddings.npy").astype('float32')
    index = faiss.read_index("faiss.index")

    texts = [chunk["text"] for chunk in chunks]

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

    tokenized_corpus = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    return chunks, text_embeddings, index, tfidf_vectorizer, tfidf_matrix, bm25, texts

chunks, text_embeddings, index, tfidf_vectorizer, tfidf_matrix, bm25, texts = load_resources()

@st.cache_resource
def load_models():
    return SentenceTransformer('all-MiniLM-L6-v2')

text_model = load_models()

# ---------- RETRIEVAL FUNCTIONS ----------

def search_faiss(query, top_k=3):
    query_embedding = text_model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, k=top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "start_time": chunks[idx]["start_time"],
            "text": chunks[idx]["text"],
            "score": distances[0][i]
        })
    return results

def search_tfidf(query, top_k=3):
    query_vector = tfidf_vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-top_k:][::-1]

    results = []
    for i in top_indices:
        results.append({
            "start_time": chunks[i]["start_time"],
            "text": chunks[i]["text"],
            "score": cosine_similarities[i]
        })
    return results

def search_bm25(query, top_k=3):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []
    for i in top_indices:
        results.append({
            "start_time": chunks[i]["start_time"],
            "text": chunks[i]["text"],
            "score": scores[i]
        })
    return results

def search_postgresql(query, top_k=3):
    # Embed and normalize the query
    query_embedding = text_model.encode([query])[0]
    norm = np.linalg.norm(query_embedding)
    if norm > 0:
        query_embedding = query_embedding / norm
    query_embedding_list = query_embedding.tolist()

    # Call the match_transcripts function (NOT execute_sql)
    response = supabase.rpc(
        "match_transcripts",
        {
            "query_embedding": query_embedding_list,
            "match_count": top_k
        }
    ).execute()

    results = []
    if response.data:
        for row in response.data:
            results.append({
                "start_time": row["start_time"],
                "text": row["text"],
                "score": row["distance"]
            })

    threshold = 1.0
    if results:
        is_no_answer = results[0]["score"] > threshold
    else:
        is_no_answer = True

    return results, is_no_answer


# ---------- STREAMLIT UI ----------

st.title("RAG Video Question Answering System")

st.video("https://www.youtube.com/watch?v=dARr3lGKwk8")

query = st.text_input("Ask your question about the seminar video:")

retrieval_method = st.selectbox(
    "Choose Retrieval Method:",
    ["FAISS (Semantic)", "TF-IDF (Lexical)", "BM25 (Lexical)", "PostgreSQL (Semantic)"]
)

if retrieval_method.startswith("PostgreSQL"):
    st.info("PostgreSQL will automatically use the best available index (IVFFLAT or HNSW). You don’t need to choose manually.")

if st.button("Search") and query:

    start_search = time.time()

    if retrieval_method.startswith("FAISS"):
        results = search_faiss(query)
        threshold = 1.5
        is_no_answer = results[0]["score"] > threshold

    elif retrieval_method.startswith("TF-IDF"):
        results = search_tfidf(query)
        threshold = 0.10
        is_no_answer = results[0]["score"] < threshold

    elif retrieval_method.startswith("BM25"):
        results = search_bm25(query)
        threshold = 2
        is_no_answer = results[0]["score"] < threshold

    elif retrieval_method.startswith("PostgreSQL"):
        results, is_no_answer = search_postgresql(query)

    end_search = time.time()

    retrieval_time = end_search - start_search

    st.write(f"**Retrieval Time:** {retrieval_time:.2f} seconds")

    if is_no_answer or not results:
        st.warning("The answer for your question is not present in the video.")
    else:
        best = results[0]
        st.success("**Best Match Found!**")

        st.markdown(f"**Timestamp:** {best['start_time']} seconds")
        st.markdown(f"**Answer Text:** {best['text']}")

        youtube_link = f"https://www.youtube.com/watch?v=dARr3lGKwk8&t={best['start_time']}s"
        st.markdown(f"[Watch this moment on YouTube]({youtube_link})")
