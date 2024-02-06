# pip install streamlit langchain lanchain-openai beautifulsoup4 python-dotenv chromadb lxml
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_core.documents import Document


load_dotenv()
def get_vectorstore_from_url(url):
    # Initialize the SitemapLoader
    web_path = url + '/sitemap.xml'
    sitemap_loader = SitemapLoader(web_path)

    # Crawl the website
    documents = sitemap_loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    document_chunks = text_splitter.split_documents(documents)

    # Assign a unique id to each chunk and include the URL in the text content
    document_chunks_with_ids = [
        Document(id=str(i), page_content=chunk.page_content + "\n\nSource: " + chunk.metadata['source'], metadata=chunk.metadata)
        for i, chunk in enumerate(document_chunks)
    ]

    # Create vectorstore from chunks 
    vectorStore = Chroma.from_documents(document_chunks_with_ids, OpenAIEmbeddings())

    return vectorStore

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
      ("system", "You are a customer support representative of BfMedia. You always answer about the company and the services they offer as you are a part of the company. Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input): 
   
    # Create conversation chain
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)

    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    # Extract the source URL from the context
    last_document = response['context'][-1]
    source_url = last_document.metadata['source']

    # Append the source URL to the response
    response_with_url = response['answer'] + f"\n\nSource: {source_url}"

    return response_with_url


# App config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title = "Chat with websites"


# Sidebar
with st.sidebar:
    st.header("Settings")
    website_URL = st.text_input("Website URL")

if website_URL is None or website_URL == "":
    st.info("Please enter website URL")

else:
    # Session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="I am a bot. How can I help you today?")
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_URL)
    

  
    # User input
    # document_chunks = get_vectorstore_from_url(website_URL) 
    document_chunks = st.session_state.vector_store 

    userQuery = st.chat_input("Type your message here")
    if userQuery is not None and userQuery != "":
        response = get_response(userQuery)
        # st.write(response)
        st.session_state.chat_history.append(HumanMessage(content=userQuery))
        st.session_state.chat_history.append(AIMessage(content=response))


    # Conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)




