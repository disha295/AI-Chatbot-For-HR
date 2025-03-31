import streamlit as st
from PyPDF2 import PdfReader
from langchain_core.documents.base import Document
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

# Initialize the LLM and embeddings
llm = Ollama(model="llama3")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# PDF Text Extraction
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
        separators=["\n\n", "\n", "(?<=\\. )", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Load website content
def load_website_content(websites):
    content = []
    loader = WebBaseLoader(websites)
    try:
        content = loader.load() 
    except Exception as e:
        print(f"An error occurred while loading content from {websites}: {e}")
    return content

# Split website content into chunks
def get_web_chunks(content_list):
    web_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "(?<=\\. )", " ", ""]
    )
    documents = web_splitter.split_documents(content_list)
    return documents

# Create vectorstore for document retrieval
def create_vectorstore(documents, chunks):
    all_documents = documents + chunks
    all_documents = [Document(page_content=content) if isinstance(content, str) else content for content in all_documents]
    vectorstore = Chroma.from_documents(all_documents, embeddings)
    return vectorstore

def get_context_retriever_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

# Create a Conversational RAG chain with search tool integration
def get_conversational_rag_chain(retriever_chain):
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# Get response from the LLM
def get_response(user_input, vectorstore, chat_history):
    retriever = vectorstore.as_retriever()
    conversation_rag_chain = get_conversational_rag_chain(retriever)
    
    response = conversation_rag_chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })
    return response['answer']

def main():
    st.set_page_config(page_title="Chat with websites", page_icon="🤖")    
    st.title("Chatbot")
    st.header("Chat with PDF and website using LLaMA3💁")
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Initialize vectorstore if not already done
    if 'vectorstore' not in st.session_state:
        # Default URL
        default_url = "https://www.uis.edu/hr"
        website_contents = load_website_content([default_url])
        documents = get_web_chunks(website_contents)
        st.session_state.vectorstore = create_vectorstore(documents, [])
    
    # User input for the chatbot
    user_question = st.text_input("Ask a Question from the PDF Files or website")

    if st.button("Send"):
        response = get_response(user_question, st.session_state.vectorstore, st.session_state.chat_history)
        st.write("AI:", response)
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=response))
    
    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            st.write("AI:", message.content)
        elif isinstance(message, HumanMessage):
            st.write("Human:", message.content)
            
    # Sidebar for uploading PDFs and entering website URLs
    with st.sidebar:
        st.title("Menu:")
        st.subheader("Enter the URLs of the websites (one URL per line):")
        website_urls = st.text_area("Paste or type the URLs here:", height=200)
        website_urls_list = website_urls.split('\n')
        websites = [url.strip() for url in website_urls_list if url.strip()]
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Process the websites and PDFs
                website_contents = load_website_content(websites)
                documents = get_web_chunks(website_contents)
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                
                # Update and store the vectorstore
                st.session_state.vectorstore = create_vectorstore(documents, text_chunks)
                st.success("Processing done! You can now ask questions.")

if __name__ == "__main__":
    main()