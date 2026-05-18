import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langfuse.langchain import CallbackHandler
from qdrant_client import QdrantClient

load_dotenv()

print("Initializing components...")

langfuse_handler = CallbackHandler()

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="qwen3:latest")

qdrant_client = QdrantClient(url="http://localhost:6333")
vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name=os.environ["QDRANT_COLLECTION_NAME"],
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ============================================================================
# IMPLEMENTATION 1: Without LCEL (Simple Function-Based Approach)
# ============================================================================
def retrieval_chain_without_lcel(query: str):
    docs = retriever.invoke(query, config={"callbacks": [langfuse_handler]})
    context = format_docs(docs)
    messages = prompt_template.format_messages(context=context, question=query)
    response = llm.invoke(messages, config={"callbacks": [langfuse_handler]})
    return response.content


# ============================================================================
# IMPLEMENTATION 2: With LCEL (LangChain Expression Language) - BETTER APPROACH
# ============================================================================
def create_retrieval_chain_with_lcel():
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


if __name__ == "__main__":
    print("Retrieving...")

    query = "what is Pinecone in machine learning?"

    # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke(
        [HumanMessage(content=query)],
        config={"callbacks": [langfuse_handler]},
    )
    print("\nAnswer:")
    print(result_raw.content)

    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

    # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke(
        {"question": query},
        config={"callbacks": [langfuse_handler]},
    )
    print("\nAnswer:")
    print(result_with_lcel)

    from langfuse import get_client
    get_client().flush()
