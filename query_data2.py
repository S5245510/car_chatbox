import argparse
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a knowledgeable car specialist. Answer the question based only on the following context:

{context}

---

User's Question: {question}

---

Provide a clear and comprehensive answer, including any relevant troubleshooting steps or maintenance tips.
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(initial_query: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Maintain context history
    context_history = ""

    while True:
        # Search the DB.
        results = db.similarity_search_with_score(initial_query, k=5)

        # Build context from results and history
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        full_context = context_history + "\n\n---\n\n" + context_text if context_history else context_text
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=full_context, question=initial_query)

        # Generate response
        model = Ollama(model="llama3.1:8b")
        response_text = model.invoke(prompt)

        sources = [doc.metadata.get("id", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        print(formatted_response)

        # Update context history
        context_history += f"User's Question: {initial_query}\nResponse: {response_text}\n"

        # Ask user for follow-up question
        follow_up = input("Do you have a follow-up question? (yes/no): ").strip().lower()
        if follow_up == 'yes':
            initial_query = input("Please enter your follow-up question: ")
        else:
            break


if __name__ == "__main__":
    main()