Car Information RAG Chatbot
Ever wished you could just ask questions about your car manual and get direct answers? This project does exactly that! It's a smart chatbot that can read through your car documents (like PDFs), understand them, and then answer your questions using the information it finds. Think of it as your personal car information expert.

This project implements a Retrieval Augmented Generation (RAG) chatbot designed to answer questions about car information based on a collection of PDF documents (e.g., owner's manuals, service guides). It combines a vector database (Chroma) for efficient document retrieval with a pluggable Large Language Model for generating context-aware answers.

Features
Document Ingestion: Easily load and process PDF documents into a searchable knowledge base.

Vector Database: Utilizes Chroma to store document embeddings for fast and relevant retrieval.

AI Language Model Integration: Designed for flexible integration with various Large Language Models (LLMs) and embedding models, with Ollama (mxbai-embed-large for embeddings, llama3.1:8b for response generation) provided as the default setup.

Intuitive Web GUI: A simple and responsive web interface for interacting with the chatbot.

Source Citation: Provides references (document and page) for the information retrieved.


Setup and Installation
Prerequisites
Python 3.8+

pip (Python package installer)

Ollama (for default setup): Ensure Ollama is installed and running on your system. You will need to pull the mxbai-embed-large and llama3.1:8b (or mistral for test_rag.py) models.

ollama pull mxbai-embed-large
ollama pull llama3.1:8b
ollama pull mistral # Only if you plan to run test_rag.py

If you plan to use other LLMs (e.g., OpenAI, Google Gemini, Anthropic Claude), ensure you have the necessary API keys and access.

Steps
Clone the Repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

(Replace your-username/your-repo-name.git with your actual GitHub repository URL.)

Install Python Dependencies:
All required Python packages are listed in requirements.txt.

pip install -r requirements.txt

Note: You may encounter deprecation warnings from LangChain, but the current code should still function. It's recommended to update to the latest import paths as suggested by the warnings for future compatibility.

Place Your Documents:
Put your PDF documents (e.g., car manuals) into the data/ directory.

Populate the Vector Database:
Run the populate_database.py script to process your documents and create the Chroma vector database.

python populate_database.py --reset

The --reset flag will clear any existing database before populating it, ensuring a fresh start.

Start the Backend API:
Open a new terminal window and start the Flask development server.

python app.py

This will start the API server, typically at http://127.0.0.1:5000/. Keep this terminal running while using the chatbot.

Access the Frontend GUI:
Open the chatbot_gui.html file in your web browser.

# On Windows (from project root):
start chatbot_gui.html
# Or, navigate to the file in your file explorer and double-click it.

Your chatbot interface should now be visible and ready for interaction.

Customizing Language Models
This RAG system is designed to be flexible with the Large Language Model (LLM) and embedding model it uses. By default, it's set up for Ollama.

1. Modifying the Embedding Function (get_embedding_function.py)
To change the embedding model, you'll modify get_embedding_function.py.

Current (Ollama):

# get_embedding_function.py
from langchain_ollama import OllamaEmbeddings
# from langchain_community.embeddings.bedrock import BedrockEmbeddings # Example for Bedrock

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return embeddings

To use a different model (e.g., Bedrock Embeddings):

You would uncomment or add the relevant import and initialization. For example, to switch to AWS Bedrock Embeddings (assuming you have AWS credentials configured):

# get_embedding_function.py
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
# from langchain_openai import OpenAIEmbeddings # For OpenAI embeddings

def get_embedding_function():
    # Option 1: Using Ollama (default)
    # embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    # Option 2: Using AWS Bedrock Embeddings
    embeddings = BedrockEmbeddings(
        credentials_profile_name="default", # Or your specific profile
        region_name="us-east-1" # Or your specific region
    )

    # Option 3: Using OpenAI Embeddings (requires API key)
    # embeddings = OpenAIEmbeddings(openai_api_key="YOUR_OPENAI_API_KEY")

    return embeddings

Steps to change:

Install the necessary LangChain integration package for your desired embedding provider (e.g., pip install langchain-aws for Bedrock, pip install langchain-openai for OpenAI).

Import the correct embedding class.

Comment out the current OllamaEmbeddings line.

Uncomment or add the line for your chosen embedding model and configure it with any required API keys or parameters.

2. Modifying the Language Model (query_data.py)
To change the LLM used for generating responses, you'll modify query_data.py.

Current (Ollama):

# query_data.py
from langchain_community.llms.ollama import Ollama
# ... other imports ...

def query_rag(query_text: str):
    # ... database preparation ...
    
    model = Ollama(model="llama3.1:8b") # Current Ollama model
    response_text = model.invoke(prompt)
    
    # ... rest of the function ...

To use a different LLM (e.g., Google Generative AI / Gemini):

You would import the relevant LLM class and initialize it.

# query_data.py
# from langchain_community.llms.ollama import Ollama # Original Ollama import
# from langchain_openai import ChatOpenAI # For OpenAI Chat models
from langchain_google_genai import ChatGoogleGenerativeAI # For Google Gemini

# ... other imports ...

def query_rag(query_text: str):
    # ... database preparation ...

    # Option 1: Using Ollama (default)
    # model = Ollama(model="llama3.1:8b")

    # Option 2: Using Google Generative AI (requires API key)
    # Ensure GOOGLE_API_KEY environment variable is set
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

    # Option 3: Using OpenAI GPT-4 (requires API key)
    # model = ChatOpenAI(model="gpt-4", openai_api_key="YOUR_OPENAI_API_KEY")

    response_text = model.invoke(prompt)

    # ... rest of the function ...

Steps to change:

Install the necessary LangChain integration package for your desired LLM provider (e.g., pip install langchain-google-genai for Gemini, pip install langchain-openai for OpenAI).

Import the correct LLM class.

Comment out the current Ollama line.

Uncomment or add the line for your chosen LLM and configure it with any required API keys or parameters. API keys are often best managed using environment variables (e.g., os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY" or setting it in your shell).

Usage
Type your car-related questions into the input field of the chatbot GUI and press "Send" or Enter. The chatbot will retrieve relevant information from your documents and provide an answer along with the sources.



This script will test predefined questions against expected responses to evaluate the system's accuracy.

Contributing
Feel free to fork this repository, submit pull requests, or open issues for any bugs or feature requests.

License
This project is open-source and available under the MIT License.