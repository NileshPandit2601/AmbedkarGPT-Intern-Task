from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import os

# ---------------------------
# Configuration
# ---------------------------
SPEECH_FILE = "speech.txt"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral"


# ---------------------------
# 1. Load text file
# ---------------------------
loader = TextLoader(SPEECH_FILE, encoding="utf-8")
docs = loader.load()


# ---------------------------
# 2. Split into chunks
# ---------------------------
text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(docs)


# ---------------------------
# 3. Create vector DB with embeddings
# ---------------------------
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vectordb = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)
vectordb.persist()


# ---------------------------
# 4. Create retriever
# ---------------------------
retriever = vectordb.as_retriever(search_kwargs={"k": 4})


# ---------------------------
# 5. Create LLM
# ---------------------------
llm = Ollama(model=OLLAMA_MODEL, temperature=0.0)


# ---------------------------
# 6. Build LCEL retrieval chain
# ---------------------------
prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the user's question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}
""")

chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ---------------------------
# 7. Interactive CLI
# ---------------------------
print("AmbedkarGPT — Retrieval QA (type 'exit' to quit)")

while True:
    query = input("\nAsk a question: ")

    if query.strip().lower() in ("exit", "quit"):
        print("Goodbye!")
        break

    answer = chain.invoke(query)
    print("\nAnswer:\n", answer)
