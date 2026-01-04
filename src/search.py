from src.embedding import EmbeddingPipeline
from src.vectorstore import FAISSStore
from groq import Groq
import os

class RAGSearch:
    def __init__(self):
        self.embedder = EmbeddingPipeline()
        self.store = FAISSStore(dim=384)
        self.llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def search_and_summarize(self, query: str):
        query_emb = self.embedder.model.encode(query)
        docs = self.store.search(query_emb)

        context = "\n".join(d.page_content for d in docs)

        prompt = f"""
        Answer the question using the context below.
        Context:
        {context}

        Question: {query}
        """

        response = self.llm.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content
