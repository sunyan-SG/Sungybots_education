from PyPDF2 import PdfReader
#from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
#from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
#from langchain.memory import ConversationMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler
#from langchain_community.llms import Bedrock
#from langchain_community.chat_models import BedrockChat
from langchain_openai import OpenAI, ChatOpenAI
from langchain.schema import ChatMessage
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import os
import sys
import io

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

#os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
st.session_state.openai_key = os.getenv('OPENAI_API_KEY')

st.set_page_config(page_title="AI Chatbot for custo",
                   layout="wide",
                   page_icon=":books::parrot:")

#st.title("AI Chatbot for Customer Services")
st.title("新国大EMBA招生查询")

# Create a container to display the chatbot's responses
stream_handler = StreamHandler(st.empty())


if "langchain_messages" not in st.session_state:
    st.session_state["langchain_messages"] = []
# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")

if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you today? 您好，请问有什么可以帮忙的？")

st.sidebar.markdown(
    """
    ### Message from SungyBots:
    Very welcome to try the chatbot developed by SungyBots 👏👏!!
    
    欢迎咨询新国大的EMBA项目 👏👏！
    
    Any further queries, feel free contact us: contact@SungyBots.com
    """
)

# retriever = None
# st.session_state.vectorDB = None
# st.session_state.pdf_docs = None

# Define conversation_chain after vectorDB is set
memory = ConversationBufferMemory(
    memory_key="chat_history", input_key="question", return_messages=False)
    #memory_key="chat_history",chat_memory=msgs,return_messages=True)


current_directory = os.getcwd()
path = current_directory+ '/' + 'vectorDB'
#path = './' + 'vectorDB'
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'})
# embeddings = OpenAIEmbeddings()
st.session_state.vectorDB = FAISS.load_local(path, embeddings)
#st.success('Ready to start!', icon="✅")

# # Debug: Print the status of OpenAI key and vectorDB
# st.write("OpenAI key:", st.session_state.get("openai_key"))
# st.write("VectorDB:", st.session_state.vectorDB)

if st.session_state.vectorDB is not None:
    vectorDB = st.session_state.vectorDB
    if hasattr(vectorDB, 'as_retriever'):
        # retriever: top 3 retrieved similarity search
        retriever = vectorDB.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3})

        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            # model_name="text-davinci-003",
            openai_api_key=st.session_state.openai_key,
            temperature=0,
            streaming=True,
            verbose=False,
            #callbacks=[stream_handler],  ## this one must be removed
        )

        st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type="stuff",
            # vectorstore=vectorDB,
            retriever=retriever,
            #verbose=False,
            verbose = False,
            memory=memory,
        )

# # Debug: Print the conversation_chain
# st.write("conversation_chain:", st.session_state.conversation_chain)

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

prompt = st.chat_input("Your question here")
if prompt:
    if st.session_state.openai_key is None:
        st.error("Please submit your OpenAI API key before using the chatbot.")
    elif st.session_state.vectorDB is None:
        st.error("Please upload PDF files and click 'Process' before using the chatbot.")
    else:
        response = st.session_state.conversation_chain.invoke(
            {'question': prompt,
             'chat_history': msgs.messages,
             }
        )
        # qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff',
        #                                  retriever=retriever)
        # response= qa.invoke(prompt)
        # ## above response type generated by RetrievalQA is different from conversation_chain

        st.chat_message("user").write(prompt)
        #st.chat_message("ai").write(response)
        st.chat_message("ai").write(response["answer"])
        msgs.add_user_message(prompt)
        msgs.add_ai_message(response["answer"])

