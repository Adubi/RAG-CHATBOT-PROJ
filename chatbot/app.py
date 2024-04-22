from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

## PROMPT TEMPLATE

prompt = ChatPromptTemplate.from_messages(

    [
        ("system", "You are a helpful assistant. Please respond to the user queries"),
        ("user", "Question:{question}")


    ]


)



## streamlit framework

st.title('Langchain Chatbot')
input_text = st.text_input('Search the topic you want')

#openAI LLM

llm= ChatOpenAI(model="gpt-3.5-turbo")
outpt_parser = StrOutputParser()
chain= prompt|llm|outpt_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))






def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = OpenAIChat(model_name="gpt-3.5-turbo", api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question, api_key):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization= True)
    docs = new_db.similarity_search_with_score(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])