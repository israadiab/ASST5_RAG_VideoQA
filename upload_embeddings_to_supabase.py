import json
import numpy as np
from supabase import create_client, Client

# Supabase credentials
url = "https://fargcblkexcrmfiztvkz.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZhcmdjYmxrZXhjcm1maXp0dmt6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDYzNTkwNTAsImV4cCI6MjA2MTkzNTA1MH0.ksSgFWTY-F8O_qIGYLTFSx7EQm74bIc6c7nsr43Nd2k"
supabase: Client = create_client(url, key)

# 1️. Load the transcript chunks and embeddings
with open("transcript.json", "r") as f:
    chunks = json.load(f)

text_embeddings = np.load("text_embeddings.npy")

print(f"Loaded {len(chunks)} chunks and {len(text_embeddings)} embeddings.")

# 2️. Upload each chunk + embedding to Supabase
for i, chunk in enumerate(chunks):
    start_time = chunk["start_time"]
    text = chunk["text"]
    embedding = text_embeddings[i].tolist()  # convert numpy array to list

    row = {
        "start_time": start_time,
        "text": text,
        "embedding": embedding
    }

    response = supabase.table("transcript_chunks").insert([row]).execute()

    if response.data:
        print(f"Inserted chunk {i}")
    else:
        print(f"Failed to insert chunk {i}: {response}")

print("All data uploaded successfully.")
