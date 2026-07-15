from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_document(file_path: str):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    return loader.load()


# chunking
def split_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


#embedding
def create_vectorstore(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def save_vectorstore(vectorstore, path="faiss_index"):
    vectorstore.save_local(path)
    print(f"Vectorstore saved to {path}/")



if __name__ == "__main__":
    with open("test.txt", "w") as f:
        f.write("""Retrieval Augmented Generation (RAG) is a technique that combines information retrieval with language generation.
It works by first retrieving relevant documents from a knowledge base, then passing them to a language model as context.
This allows the model to answer questions grounded in specific documents rather than relying purely on training data.
RAG is especially useful for question answering over private or frequently updated documents.
The retrieval step uses vector similarity search to find chunks most relevant to the user query.""")

    docs = load_document("test.txt")
    chunks = split_documents(docs)
    vectorstore = create_vectorstore(chunks)
    save_vectorstore(vectorstore)

    results = vectorstore.similarity_search("how does retrieval work?", k=2)
    print(f"Query: 'how does retrieval work?'")
    print(f"Top {len(results)} results:\n")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i} ---")
        print(doc.page_content)