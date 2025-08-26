from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import openai 
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="env_variable.env")

DATA_PATH = "data"
DATABASE_PATH = "database"

openai.api_key = os.environ['OPENAI_API_KEY']

def main():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_db(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)
    return chunks


def save_to_db(chunks: list[Document]):
    if os.path.exists(DATABASE_PATH):
        print("DB already exists. Delete it first if you want to recreate it.")
        return  

    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(chunks, embeddings, persist_directory=DATABASE_PATH)
    db.persist()
    print(f"Saved {len(chunks)} chunks to {DATABASE_PATH}.")

if __name__ == "__main__":
    main()