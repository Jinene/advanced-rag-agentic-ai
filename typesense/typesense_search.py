import typesense
from sentence_transformers import SentenceTransformer

client = typesense.Client({
    "nodes": [{"host": "localhost", "port": "8108", "protocol": "http"}],
    "api_key": "xyz",
    "connection_timeout_seconds": 2
})

model = SentenceTransformer("all-MiniLM-L6-v2")

def search(query):
    emb = model.encode(query).tolist()
    return client.collections["books"].documents.search({
        "q": "*",
        "vector_query": f"embedding:([{','.join(map(str, emb))}], k:3)"
    })
