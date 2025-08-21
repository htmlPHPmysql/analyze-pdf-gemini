import streamlit as st
import pdfplumber
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import asyncio

# A list of available models for the user to select
AVAILABLE_MODELS = [    
    "gemini-2.5-flash-lite",
    "gemma-3n-e2b-it",
    "gemma-3n-e4b-it",
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemma-3-4b-it",
    "gemma-3-1b-it",
    "learnlm-2.0-flash-experimental",
    "gemini-2.5-pro-preview-tts",
    "gemini-2.5-flash-preview-tts",
    "gemini-2.0-flash-thinking-exp-1219",
    "gemini-2.0-flash-thinking-exp",
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-exp-1206",
    "gemini-2.0-pro-exp-02-05",
    "gemini-2.0-pro-exp",
    "gemini-2.0-flash-lite-preview"
]

def get_pdf_text(pdf_documents):
    """
    Extracts and concatenates text from a list of PDF documents.
    """
    text = ""
    for pdf in pdf_documents:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    """
    Splits a long string of text into smaller, overlapping chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """
    Creates a FAISS vector database from text chunks using Gemini embeddings.
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def create_vectorstore_async():
            # Use a supported embedding model
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore = FAISS.from_texts(
                texts=text_chunks,
                embedding=embeddings
            )
            return vectorstore

        vectorstore = loop.run_until_complete(create_vectorstore_async())
        return vectorstore

    except Exception as e:
        st.error(f"Error creating the vector database: {e}")
        return None
    finally:
        if loop and not loop.is_running():
            loop.close()

def get_conversation_chain(vectorstore, model_name):
    """
    Initializes and returns a conversational chain for the chat assistant.
    """
    try:
        # Use the model selected by the user
        llm = ChatGoogleGenerativeAI(model=model_name)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question"
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        prompt = PromptTemplate.from_template(
            """
            Below are document fragments and chat history.
            Answer the user's question ONLY based on the following document fragments and chat history.

            If the answer is NOT in the documents, respond only with:
            "Unfortunately, I could not find the answer to this question in the provided documents."

            Do not use any knowledge from outside the documents.
            Do not guess. Do not create an answer if you are not certain based on the documents.

            Chat history:
            {chat_history}

            Context from documents:
            {context}

            User's question.:
            {question}

            Answer:
            """
        )
        conversation_chain = (
            {"context": retriever,
             "question": RunnablePassthrough(),
             "chat_history": lambda x: memory.load_memory_variables({})["chat_history"]}
            | prompt
            | llm
            | StrOutputParser()
        )
        st.session_state.memory = memory
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating the conversation chain: {e}")
        return None

def handle_userinput(user_question):
    """
    Processes a user's question and updates the chat history with the response.
    """
    if "conversation" not in st.session_state or not st.session_state.conversation:
        st.error("Please process your documents first.")
        return

    try:
        response = st.session_state.conversation.invoke(user_question)
        st.session_state.memory.save_context(
            {"question": user_question},
            {"output": response}
        )
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "ai", "content": response})
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    except Exception as e:
        # Check for specific rate-limit errors
        if "429" in str(e):
            st.error(f"Quota exceeded for the selected model. Please try again in a minute or select a different model.")
        else:
            st.error(f"Error while processing the request: {e}")

def main():
    """
    The main function that runs the Streamlit application.
    """
    load_dotenv()
    st.set_page_config(page_title="Ask question about your PDFs")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "memory" not in st.session_state:
        st.session_state.memory = None

    st.header("Documents AI Assistant")
    user_question = st.chat_input("Ask question about your file...")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_documents = st.file_uploader(
            "Upload your documents and click on 'Process'", accept_multiple_files=True
        )
        
        # User can select the model from a dropdown
        selected_model = st.selectbox(
            "Select AI Model:",
            options=AVAILABLE_MODELS,
            key="selected_model"
        )

        if st.button("Process"):
            if not pdf_documents:
                st.error("Upload at least one doc")
                return
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_documents)
                if not raw_text:
                    st.error("No text found in the uploaded documents.")
                    return
                else:
                    st.success("Text extracted successfully!")

                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                if vectorstore is None:
                    return

                # Pass the selected model to the conversation chain function
                st.session_state.conversation = get_conversation_chain(vectorstore, selected_model)
                if st.session_state.conversation is None:
                    return

                st.success(f"Documents {len(pdf_documents)} processed successfully")
                st.session_state.chat_history = []

if __name__ == '__main__':
    main()