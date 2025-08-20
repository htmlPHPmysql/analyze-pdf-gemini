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

# Added asyncio to manage the event loop
import asyncio

# A dictionary to store a cache for the event loop
loop = None

def get_pdf_text(pdf_documents):
    """
    Extracts and concatenates text from a list of PDF documents.

    This function iterates through a list of PDF file-like objects, opens each
    one using pdfplumber, and extracts the text from every page. The text from
    all documents is then combined into a single string.

    Args:
        pdf_documents (list): A list of file-like objects representing the
                              uploaded PDF documents.

    Returns:
        str: A single string containing all the extracted text from the
             documents. Returns an empty string if no text is found.
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

    The text is split using a character-based splitter. This is a common
    preprocessing step for large language models to ensure that the context
    can be managed effectively within a model's token limit.

    Args:
        text (str): The raw text to be split into chunks.

    Returns:
        list: A list of strings, where each string is a chunk of the original
              text.
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

    This function converts text chunks into numerical vectors using Google's
    `embedding-001` model and stores them in a FAISS index, which is optimized
    for similarity search. This allows for fast retrieval of relevant
    information.

    Args:
        text_chunks (list): A list of text strings (chunks) to be embedded
                            and stored.

    Returns:
        FAISS.vectorstore.FAISS: A FAISS vector store object, or None if
                                 an error occurs during creation.

    Raises:
        Exception: Catches and handles any exceptions that occur during the
                   vector store creation process, displaying an error message
                   to the user.
    """
    try:
        # Create a new event loop and run the embedding operation
        # This is a workaround to the 'no current event loop' error
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def create_vectorstore_async():
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
        # It's good practice to close the loop once done
        if loop and not loop.is_running():
            loop.close()

def get_conversation_chain(vectorstore):
    """
    Initializes and returns a conversational chain for the chat assistant.

    This chain uses a Retrieval-Augmented Generation (RAG) approach. It
    configures a `gemini-pro` language model, a retriever to fetch relevant
    documents from the vector store, and a prompt template to guide the model's
    response generation based on the retrieved context and chat history.

    Args:
        vectorstore (FAISS.vectorstore.FAISS): The vector store containing
                                               the document embeddings.

    Returns:
        langchain_core.runnables.Runnable: A runnable conversation chain, or
                                           None if an error occurs.

    Raises:
        Exception: Catches and handles any exceptions that occur during the
                   chain creation process, displaying an error message to
                   the user.
    """
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-flash")
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

    This function checks for the existence of the conversation chain and then
    invokes it with the user's question. It saves the context to memory and
    appends both the user's question and the AI's response to the chat history,
    which is then displayed in the Streamlit interface.

    Args:
        user_question (str): The question submitted by the user.

    Returns:
        None: The function updates the Streamlit session state and chat
              interface directly.

    Raises:
        Exception: Catches and handles any exceptions during the invocation
                   or history update, displaying an error message to the user.
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
        st.error(f"Error while processing the request: {e}")

def main():
    """
    The main function that runs the Streamlit application.

    This function sets up the Streamlit page configuration, initializes the
    session state variables, and defines the user interface layout, including
    the chat input and the sidebar for document uploading and processing. It
    orchestrates the entire application flow based on user interactions.
    """
    load_dotenv()
    st.set_page_config(page_title="Ask question about your PDFs")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "memory" not in st.session_state:
        st.session_state.memory = None

    st.header("PDF Document Assistant")
    user_question = st.chat_input("Ask question about your PDF...")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_documents = st.file_uploader(
            "Upload your documents and click on 'Process'", accept_multiple_files=True
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

                st.session_state.conversation = get_conversation_chain(vectorstore)
                if st.session_state.conversation is None:
                    return

                st.success(f"Documents {len(pdf_documents)} processed successfully")
                st.session_state.chat_history = []

if __name__ == '__main__':
    main()