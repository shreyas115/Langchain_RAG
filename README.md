# 📘 Retrieval-Augmented Generation (RAG) with LangChain  

This project demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline using [LangChain](https://www.langchain.com/) and OpenAI embeddings. It shows how to build a vector database from documents and then query it with an LLM to generate context-aware answers.  

---

## 📂 Main elements of the repository 

- **`create_db.py`**  
  - Loads documents from a given path.  
  - Splits documents into smaller chunks for better retrieval.  
  - Creates embeddings for each chunk.  
  - Stores the chunks and embeddings into a persistent vector database.
  
- **`data`** - Contains the data that is going to be used in this project. You can replace it with your own data source.

- **`query_database.py`**  
  - Prompts the user for a query.  
  - Creates embeddings for the query using OpenAI.  
  - Retrieves the **k most similar documents** from the vector database built by `create_db.py`.  
  - Passes the retrieved context to a Large Language Model (LLM).  
  - Returns an answer that is retrieved from the most relevant documents.  

- **`requirements.txt`** – Project dependencies.  

---

## ⚙️ Setup  

1. **Clone this repository**  
   git clone https://github.com/shreyas115/Langchain_RAG.git

   cd Langchain_RAG

3. **Install dependencies**  
   pip install -r requirements.txt

4. **Set your API key**
   
   Create a .env file in the project root with:
   
   **OPENAI_API_KEY=your_key_here***

## 🚀 Usage
1. **Create the Vector Database**

    Run the following to load documents, split them into chunks, and store embeddings in a vector database:

    **python create_db.py** 

2. **Query the Database**

    Ask a question and retrieve context-aware answers from the LLM:
    
    **python query_database.py "Type the question here"**


