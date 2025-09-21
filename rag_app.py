
"""
Milvus RAG (Retrieval-Augmented Generation) Application

This script builds a RAG pipeline that uses a local directory of text files as its knowledge base.
The pipeline performs the following steps:
1. Loads documents from a specified data directory.
2. Splits the documents into smaller, manageable chunks.
3. Generates embeddings (numerical representations) for each chunk using an OpenAI model.
4. Stores these chunks and their embeddings in a Milvus vector database.
5. Creates a LangChain retrieval chain that can answer questions based on the stored documents.
"""

# --- 1. IMPORTS AND ENVIRONMENT SETUP ---
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from a .env file (especially the OPENAI_API_KEY)
load_dotenv()

# Set the OpenAI API key in the environment
# This is required by the LangChain library to authenticate with OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")
os.environ["OPENAI_API_KEY"] = api_key

# --- 2. DATA LOADING AND PREPARATION ---

# Define the directory containing the source documents
DATA_DIR = 'data'

# List all files in the data directory
try:
    files = os.listdir(DATA_DIR)
    print(f"📚 Found {len(files)} files in the '{DATA_DIR}' directory.")
except FileNotFoundError:
    raise FileNotFoundError(f"The data directory '{DATA_DIR}' was not found. Please create it and add your text files.")

# Initialize a list to hold the document chunks
document_chunks = []

# Initialize a text splitter to break down large documents
# chunk_size: The maximum number of characters in a chunk.
# chunk_overlap: The number of characters to overlap between chunks to maintain context.
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512, 
    chunk_overlap=64
)

# Process each file in the data directory
for file_name in files:
    file_path = os.path.join(DATA_DIR, file_name)
    try:
        with open(file_path, encoding='utf-8', errors='ignore') as f:
            file_content = f.read()
        
        # Split the file content into smaller text chunks
        chunks = text_splitter.split_text(file_content)
        
        # Create a Document object for each chunk with associated metadata
        for i, chunked_text in enumerate(chunks):
            document_chunks.append(Document(
                page_content=chunked_text,
                metadata={"doc_title": os.path.splitext(file_name)[0], "chunk_num": i}
            ))
        print(f"   - Processed '{file_name}' into {len(chunks)} chunks.")
    except Exception as e:
        print(f"⚠️ Could not process file {file_name}: {e}")

if not document_chunks:
    raise ValueError("No documents were successfully loaded or chunked. Check the data directory and file contents.")

# --- 3. VECTOR STORE AND RETRIEVER SETUP ---

# Initialize the OpenAI embeddings model
# This model converts text chunks into high-dimensional vectors (embeddings).
print("🧠 Initializing OpenAI embeddings model...")
embeddings = OpenAIEmbeddings()

# Configure the connection to the Milvus vector database
# The script detects if it's running in a Docker container and uses the appropriate host.
milvus_host = os.getenv('MILVUS_HOST', 'localhost')
milvus_port = os.getenv('MILVUS_PORT', '19530')
milvus_uri = f'http://{milvus_host}:{milvus_port}'

print(f"🔗 Connecting to Milvus at: {milvus_uri}")

# Create the Milvus vector store
# This command sends the document chunks and their embeddings to Milvus for storage and indexing.
# `from_documents` is a convenient way to populate the database in one go.
vector_store = Milvus.from_documents(
    documents=document_chunks,
    embedding=embeddings,
    connection_args={"uri": milvus_uri},
    collection_name="wimpy_kid_rag_collection"  # A descriptive name for the collection in Milvus
)

print("✅ Milvus vector store created and populated.")

# Create a retriever from the vector store
# The retriever is the component responsible for fetching relevant documents from the vector store
# based on a user's query.
retriever = vector_store.as_retriever()
print("🔍 Retriever created.")

# --- 4. LANGCHAIN RAG CHAIN CONSTRUCTION ---

# Define the prompt template for the RAG chain
# This template structures the final prompt sent to the language model.
# - {context}: This will be filled with the relevant documents fetched by the retriever.
# - {question}: This will be the user's original question.
RAG_PROMPT_TEMPLATE = """
You are a helpful assistant who is an expert on the "Diary of a Wimpy Kid" book series.
Answer the user's question based only on the following context.
If the context does not contain the answer, state that you don't have enough information.

CONTEXT:
{context}

QUESTION:
{question}
"""

prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# Initialize the language model (LLM)
# We use ChatOpenAI for a conversational, chat-based interaction.
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# Create the RAG chain using LangChain Expression Language (LCEL)
# This declarative pipeline defines the flow of data.
#
# The `|` (pipe) operator chains components together.
#
# The chain works as follows:
# 1. An input dictionary with "context" and "question" is created in parallel.
#    - "context": The `retriever` is called with the user's question, fetching relevant documents.
#    - "question": The user's question is passed through unchanged using `RunnablePassthrough`.
# 2. The resulting dictionary is passed to the `prompt` template.
# 3. The formatted prompt is passed to the `llm` for processing.
# 4. The `StrOutputParser` extracts the string content from the LLM's response.

print("⛓️ Assembling the RAG chain...")
chain = (
    RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough()
    })
    | prompt
    | llm
    | StrOutputParser()
)
print("✅ RAG chain is ready!")

# --- 5. EXAMPLE INVOCATION (for direct script execution) ---

if __name__ == "__main__":
    print("--- Testing RAG Chain ---")
    
    question1 = "What is the main character's name?"
    print(f"❓ Question: {question1}")
    response1 = chain.invoke(question1)
    print(f"💬 Answer: {response1}")

    question2 = "Who is Rowley Jefferson?"
    print(f"❓ Question: {question2}")
    response2 = chain.invoke(question2)
    print(f"💬 Answer: {response2}")

    # This question is outside the context of the books
    question3 = "What is the capital of France?"
    print(f"❓ Question: {question3}")
    response3 = chain.invoke(question3)
    print(f"💬 Answer: {response3}")
    print("--- Test Complete ---")

#IMPORTS

from langchain.chains import LLMChain  # Import LLMChain for chaining language model operations
from langchain.prompts import PromptTemplate  # Import PromptTemplate for creating structured prompts
from langchain_openai import OpenAI  # Import OpenAI class for connecting to OpenAI's language models
from dotenv import load_dotenv  # Import load_dotenv to load environment variables from .env file
import os  # Import os module for operating system interface functions
from langchain_core.runnables import RunnableParallel

load_dotenv()  # Load environment variables from .env file

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Set the OpenAI API key from environment variables

# Classes and functions will go here

template = "You are a helpful assistant that understand concepts and find information for 5 year old kid.  Question: {question}" #Defines the AI's role and behavior

prompt = PromptTemplate.from_template(template)  # Create a prompt template from the defined template

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")  # Initialize the language model with specified parameters  

from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus

embeddings = OpenAIEmbeddings()  # Initialize the OpenAI embeddings model

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

from langchain.schema  import Document

files = os.listdir('data')  # List all files in the 'data' directory

file_texts = []  # Initialize a list to hold the text chunks

for file in files:
    with open(f"data/{file}", encoding='utf-8', errors='ignore') as f:
        file_text = f.read()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=64,
    )
    texts = text_splitter.split_text(file_text)
    for i, chunked_text in enumerate(texts):
        file_texts.append(Document(page_content=chunked_text,
        metadata={"doc_title": file.split(".")[0], "chunk_num": i}))

import os

# Detect if running in Docker container
milvus_host = os.getenv('MILVUS_HOST', 'localhost')
milvus_port = os.getenv('MILVUS_PORT', '19530')
milvus_uri = os.getenv('MILVUS_URI', f'http://{milvus_host}:{milvus_port}')

print(f"🔗 Connecting to Milvus at: {milvus_host}:{milvus_port}")
print(f"🔗 Using URI: {milvus_uri}")

vector_store = Milvus.from_documents(
    file_texts,
    embeddings,                 
    connection_args={"uri": milvus_uri},
    collection_name="wimpy_kid"
)  # Create a Milvus vector store object from the document chunks

retriever = vector_store.as_retriever()  # Convert the vector store to a retriever interface

template = """Answer based on the context {context}. Question: {question}"""  # Define a new prompt template with context

prompt = PromptTemplate.from_template(template)  # Create a prompt template from the defined template

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers  import StrOutputParser

# RAG CHAIN CONSTRUCTION

# Step 1: Create RunnableParallel - runs two operations simultaneously
runnable = RunnableParallel({
    "context": retriever,              # Example: Searches for "Greg is the main character of Diary of a Wimpy Kid..."
    "question": RunnablePassthrough()  # Example: Passes "Who is Greg Heffley?" unchanged
})
# Example output: {"context": "Greg is the main character...", "question": "Who is Greg Heffley?"}

# Step 2: Create the complete RAG chain
chain = runnable | prompt | llm | StrOutputParser()

# CHAIN FLOW EXAMPLES:
# Input: "Who is Greg Heffley?"
# 
# runnable: 
#   ↓ Processes input in parallel:
#   - context: retriever searches Milvus → ["Greg is the main character of Diary of a Wimpy Kid series...", "He is in middle school..."]
#   - question: RunnablePassthrough() → "Who is Greg Heffley?"
#   ↓ Output: {"context": "Greg is the main character...", "question": "Who is Greg Heffley?"}
#
# prompt:
#   ↓ Takes the dictionary and fills template:
#   ↓ Output: "Answer based on the context Greg is the main character of Diary of a Wimpy Kid series... Question: Who is Greg Heffley?"
#
# llm:
#   ↓ ChatGPT processes the prompt:
#   ↓ Output: AIMessage(content="Greg Heffley is the main character in the Diary of a Wimpy Kid book series...")
#
# StrOutputParser():
#   ↓ Extracts clean text from AIMessage:
#   ↓ Final Output: "Greg Heffley is the main character in the Diary of a Wimpy Kid book series..."

# print(chain)

# Test invocations - uncomment for local testing
# print(chain.invoke("Who is Greg Heffley?"))  # Invoke the chain with a sample question
# print(chain.invoke("Who is president of USA"))  # Invoke the chain with a sample question