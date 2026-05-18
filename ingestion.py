import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langfuse.langchain import CallbackHandler

load_dotenv()

if __name__ == "__main__":
    langfuse_handler = CallbackHandler()

    print("Ingesting...")
    loader = TextLoader("mediumblog1.txt", encoding="utf-8")
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("ingesting...")
    QdrantVectorStore.from_documents(
        texts,
        embeddings,
        url="http://localhost:6333",
        collection_name=os.environ["QDRANT_COLLECTION_NAME"],
    )
    print("finish")
