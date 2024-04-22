import streamlit as st


from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
## Data Ingestion

from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

# Vector Embedding And Vector Store

from langchain_community.vectorstores import FAISS

## LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.messages import AIMessage, HumanMessage


api_key = st.text_input("Enter your OPEN AI API Key:", type="password", key="api_key_input")
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

def data_ingestion(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_text(text)
    return docs

def get_vector_store(docs):
    # Extract texts from the documents
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore_faiss=FAISS.from_texts(
        docs,
        embeddings
        
        
    )
        vectorstore_faiss.save_local('faiss_index')


prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}



Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_openai_llm():
    ##create the Anthropic Model
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", api_key=api_key)
    
    return llm


def get_response_llm(llm,vectorstore_faiss,query, chat_history):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query}, {"chat_history":chat_history})
    return answer['result']





def main():


    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]
    st.title("PDF Parser")
    
    st.header("Chat with PDF using OPEN AI GPT")
    user_question = st.chat_input("Ask a Question from the PDF Files")

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)


    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Vectors Update") and api_key:
            with st.spinner("Processing..."):
                raw_text = data_ingestion(pdf_docs)
                get_vector_store(raw_text)
                st.success("Done")
    

    if user_question is not None and user_question != "":
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        with st.chat_message("Human"):
            st.markdown(user_question)

        with st.chat_message("AI"):

            faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            llm = get_openai_llm()

            response= get_response_llm(llm, faiss_index, user_question, st.session_state.chat_history)
            st.write(response)



        st.session_state.chat_history.append(AIMessage(content=response))

if __name__ == "__main__":
    main()
