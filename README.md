# PDF Parser with Chat using OPEN AI GPT

This project allows users to interactively query information from PDF files using OpenAI's GPT models. The system provides a conversational interface where users can ask questions related to the content of the uploaded PDF documents, and the system responds with relevant answers.

## Requirements

- Python 3.7
- Streamlit
- langchain_openai
- PyPDF2

## Installation

1. Clone this repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Obtain an OpenAI API key and enter it when prompted.

## How to Use

1. Run the script using `streamlit run filename.py`.
2. Upload PDF files containing the desired information.
3. Click on the "Vectors Update" button to process the uploaded PDF files and create/update the vector store.
4. Once the vector store is updated, you can start asking questions related to the content of the PDF files in the chat interface.
5. The system will use OpenAI's GPT model to generate responses based on the provided queries and the information extracted from the PDF files.

## Components

### Data Ingestion

- PDF documents are ingested and processed to extract text using PyPDF2.

### Vector Embedding and Vector Store

- Text from the PDF documents is embedded into vectors using OpenAI's embeddings.
- Vector Store is created/updated using FAISS, enabling efficient similarity search.

### Language Models

- OpenAI's GPT-3.5 turbo model is used to provide conversational responses to user queries.
- A Prompt Template is defined to structure the input for the language model.

### User Interface

- Streamlit is used to create an interactive web application.
- Users can upload PDF files, ask questions, and receive responses in a conversational format.

## Features

- Conversational interface for querying information from PDF files.
- Automatic creation/update of vector store for efficient search.
- Integration with OpenAI's GPT-3.5 model for generating responses.

## Example Usage

1. User uploads PDF files containing business reports.
2. User asks questions such as "What is the revenue growth for Q1 2024?".
3. The system extracts relevant information from the PDF files and provides an answer.

## Contributors

- Ibrahim Adubi
