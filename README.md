# Documents-Analysis-with-Langchain-and-Chromadb
Here’s a clear, professional **README.md** file for your LangChain + ChromaDB project 👇

---

```markdown
# 🧠 PDF Knowledge Search using LangChain, Hugging Face & ChromaDB

This project demonstrates how to **extract**, **embed**, and **query** knowledge from a PDF document using **LangChain**, **Hugging Face sentence embeddings**, and a **local Chroma vector database**.

---

## 🚀 Features
✅ Load and process any PDF document  
✅ Automatically split large text into smaller chunks  
✅ Create embeddings using Hugging Face models  
✅ Store embeddings locally in ChromaDB  
✅ Perform semantic similarity search (ask questions about the PDF!)  

---

## 🏗️ Project Structure
```

📁 project/
│
├── main.py                # Your Python script (the code below)
├── chroma_db/             # Local vector database (auto-created)
├── requirements.txt       # Required dependencies
└── README.md              # Documentation (this file)

````

---

## ⚙️ Installation

### 1️⃣ Clone or download this repository
```bash
git clone https://github.com/your-username/pdf-knowledge-search.git
cd pdf-knowledge-search
````

### 2️⃣ Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate       # On Windows
# OR
source venv/bin/activate    # On macOS/Linux
```

### 3️⃣ Install dependencies

Create a `requirements.txt` file with the following packages:

```text
langchain
langchain-community
langchain-huggingface
langchain-text-splitters
chromadb
pypdf
sentence-transformers
```

Then install them:

```bash
pip install -r requirements.txt
```

---

## 📄 How It Works

### Step 1️⃣ — Import Libraries

We use LangChain’s document loaders, text splitters, embeddings, and vector stores.

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
```

---

### Step 2️⃣ — Load the PDF

Use the `PyPDFLoader` to read your document.

```python
pdf_path = r"C:\Users\Zafrullah Khan\Downloads\Generative ai and chat boat\waziristan.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()
```

---

### Step 3️⃣ — Split Text into Chunks

Large PDFs are divided into manageable chunks for better embedding performance.

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = splitter.split_documents(documents)
```

---

### Step 4️⃣ — Create Text Embeddings

Convert text chunks into vector representations using Hugging Face embeddings.

```python
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

---

### Step 5️⃣ — Store in ChromaDB

Save all embeddings locally in a **persistent vector database**.

```python
vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db")
```

---

### Step 6️⃣ — Query the Knowledge Base

Ask natural language questions to find relevant sections.

```python
query = "what is this document about?"
results = vectorstore.similarity_search(query, k=3)
```

---

### Step 7️⃣ — Display Results

Print the top 3 most relevant chunks.

```python
print("\n Top 3 relevant chunks:\n")
for i, res in enumerate(results, 1):
    print(f"Result ({i}):\n{res.page_content[:300]}...\n")
```

---

## 🧩 Example Output

```
Top 3 relevant chunks:

Result (1):
The document discusses the historical background and development of Waziristan, focusing on its geography and tribal system...

Result (2):
Waziristan’s role in regional politics and its strategic importance are examined in relation to the broader South Asian context...
```

---

## 💡 Notes

* Always use **raw strings (r"…")** for file paths on Windows.
* The **`chroma_db/`** folder is auto-created for persistent storage.
* You can replace the Hugging Face model with any other embedding model (e.g., `all-mpnet-base-v2`).

---

## 🧰 Future Improvements

* Add a **Flask/Streamlit UI** for interactive chat-based queries.
* Support **multiple PDFs** and merged knowledge bases.
* Integrate **LLMs** (e.g., Gemini, GPT, or Ollama) for richer answers.

---

## 🧑‍💻 Author

**Zafarullah Khan**
📚 AI Student | 💬 Tech Enthusiast | 💡 Learning LangChain & Generative AI

---

## 🪪 License

This project is open-source under the [MIT License](https://opensource.org/licenses/MIT).

```

---

Would you like me to make this README more **beginner-friendly with extra comments and emojis**, or keep it **professional (GitHub-style)**?
```
