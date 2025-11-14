# **AmbedkarGPT-Intern-Task**

A Python-based Retrieval-Augmented Generation (RAG) prototype developed for **Kalpit Pvt Ltd (UK) – AI Intern Hiring Assignment 1**.
This project uses LangChain, ChromaDB, HuggingFace Embeddings, and Ollama (Mistral 7B) to answer questions based solely on Dr. B. R. Ambedkar’s speech.

---

# **Project Summary**

This system ingests the provided speech, splits it into chunks, embeds those chunks, stores them in a local vector store, retrieves relevant context based on a user query, and generates precise answers—powered entirely **offline**.

Built completely with **free, open-source tools**.

---

# **Tech Stack**

| Component            | Tool                               |
| -------------------- | ---------------------------------- |
| Programming Language | **Python 3.8+**                    |
| Framework            | **LangChain**                      |
| Embeddings           | **HuggingFace – all-MiniLM-L6-v2** |
| Vector Database      | **ChromaDB (local)**               |
| LLM                  | **Ollama – Mistral 7B**            |
| Output               | Text + `output.jpg`                |

---

# **Repository Structure**

```
AmbedkarGPT-Intern-Task/
│── main.py
│── speech.txt
│── output.jpg
│── requirements.txt
│── README.md
```

---

# **Output Preview**

### **Generated Output**

![Output](output.jpg)

---

# **System Architecture**

### **RAG System Architecture Diagram (Mermaid)**

```mermaid
flowchart TD

A[User Query] --> B[Retriever: ChromaDB Search]
B --> C[Relevant Chunks]

C --> D[LLM (Mistral 7B via Ollama)]
A --> D

D --> E[Final Answer]
E --> F[Output Image + Console Output]

subgraph Embedding & Storage
G[Speech Text]
G --> H[Text Splitting]
H --> I[HuggingFace Embeddings]
I --> J[ChromaDB Storage]
end
```

---

# **Installation & Setup**

## **1. Clone the Repository**

```bash
git clone https://github.com/NileshPandit2601/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task
```

## **2. Install Python Packages**

```bash
pip install -r requirements.txt
```

---

# **Ollama Setup**

## **Install Ollama**

(Linux/macOS)

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

## **Download Mistral 7B**

```bash
ollama pull mistral
```

---

# **Run the Project**

```bash
python main.py
```

This will:

* Load the speech
* Split into chunks
* Generate embeddings
* Store in ChromaDB
* Ask for your question
* Retrieve relevant chunks
* Produce LLM response
* Save output image

---

# **Example**

**User Input:**

```
What does Ambedkar identify as the root cause of caste?
```

**System Output:**

* Retrieves chunks about shastras, authority, and caste
* Generates grounded answer using Mistral 7B
* Saves final result to `output.jpg`

---

# **Features Implemented**

✔ Fully local RAG pipeline
✔ No API keys or cloud services
✔ ChromaDB for fast retrieval
✔ HuggingFace embeddings
✔ Offline LLM inference
✔ Logging + output image
✔ Clean, modular structure

---

# **Future Enhancements**

* Add GUI using Streamlit
* Expand to multiple documents
* Add caching for faster retrieval
* Add evaluation metrics (BLEU, ROUGE)
* Enhanced visualization of retrieved chunks

---

# **Acknowledgements**

This project is built as part of **Kalpit Pvt Ltd – AI Intern Hiring (Phase 1)** using open-source tools from:

* LangChain
* HuggingFace
* ChromaDB
* Ollama

---
