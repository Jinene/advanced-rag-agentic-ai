from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

class EmbeddingPipeline:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )

    def chunk_documents(self, docs):
        return self.splitter.split_documents(docs)

    def embed(self, chunks):
        texts = [c.page_content for c in chunks]
        return self.model.encode(texts, show_progress_bar=True)
