import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA

load_dotenv()

def load_vectorstore(path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def build_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain


def ask(qa_chain, question):
    response = qa_chain.invoke({"query": question})
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {response['result']}")
    print(f"\nSources:")
    for i, doc in enumerate(response['source_documents']):
        print(f"\n--- Source {i} ---")
        print(doc.page_content)


if __name__ == "__main__":
    vectorstore = load_vectorstore()
    qa_chain = build_qa_chain(vectorstore)
    ask(qa_chain, "how does retrieval work?")