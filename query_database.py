import argparse
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import openai

load_dotenv(dotenv_path="env_variable.env")
DATABASE_PATH = "chroma"
openai.api_key = os.environ['OPENAI_API_KEY']

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Create embeddings and then create a vector store
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory=DATABASE_PATH, embedding_function=embeddings)

    # Search the DB for the most relevant documents
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 1.0:
        print("No relevant documents found.")
        return
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    llm = ChatOpenAI()
    response = llm.predict(prompt)
    
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()