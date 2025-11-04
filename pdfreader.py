# 🚀 Importing libraries
from langchain_community.document_loaders import PyPDFLoader
# FIX 1: Corrected import path for text splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ---

# 📄 Step 1: Load the PDF document
# FIX 2: IMPORTANT! Use a raw string (r"...") and include the ACTUAL PDF file name.
pdf_path = r"C:\Users\Zafrullah Khan\Downloads\Generative ai and chat boat\waziristan.pdf" 
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# ---

# ✂️ Step 2: Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = splitter.split_documents(documents)

# ---

# 🧠 Step 3: Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---

# 💾 Step 4: Store in ChromaDB (local vector database)
vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db")

# ---

# 🔍 Step 5: Query example
query = "what is this document about?"
results = vectorstore.similarity_search(query, k=3)

# ---

# 🖨️ Step 6: Print results
print("\n Top 3 relevant chunks:\n")
for i, res in enumerate(results, 1):
    # Print the result index and the first 300 characters of the page content
    print(f"Result ({i}):\n{res.page_content[:300]}...\n")