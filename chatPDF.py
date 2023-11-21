# Required librairies pyPDf2 langchain python-dotenv streamli-chat
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import faiss
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message
 
 
st.set_page_config(page_title = "Chat_PDF" , page_icon = "gear" , layout = "wide")
st.snow()
 
 
# get pd extract text and divide chunks
def get_extract_chunks(pdf_docs):        
    #pdf_docs is an array of uploaded pdfs
    content = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            content += page.extract_text()
    # End extraction
 
    # start chunks breakdown
    splitter = CharacterTextSplitter(
        separator= "\n",
        chunk_size = 1400,
        chunk_overlap = 230,
        length_function = len
    )
    chunks = splitter.split_text(content)
    return chunks
 
##v create vectorestore from chunks
def create_vectorstore(chunks):
    current_embedding = OpenAIEmbeddings()
    vectorstore = faiss.FAISS.from_texts(texts=chunks,embedding=current_embedding)
    return vectorstore
 
def get_conversation_chain(vectorestore):
    current_llm = ChatOpenAI()
    current_memory = ConversationBufferMemory(
        memory_key= 'chat_history',
        return_messages= True,
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = current_llm,
        retriever = vectorestore.as_retriever(),
        memory = current_memory
    )
    return conversation_chain
 
def get_user_input(user_input):
    if not st.session_state.pdf_processed:
        st.info("Please Upload  PDF files")
        return
    answer = st.session_state.conversation({'question':user_input})
    st.session_state.chat_history = answer['chat_history']
    for idx , value in enumerate(st.session_state.chat_history):
        if idx % 2 == 0:
            message(value.content, is_user=True, key=str(idx) + '_user')
        else:
            message(value.content, key=str(idx))
           
 
 
def main():
    load_dotenv()
    st.header("Hey, chat locally here ! :question:")
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    user_input = st.text_input("Please Enter your Question ")
    if user_input:
        get_user_input(user_input)
    with st.sidebar:
        st.subheader("Please upload PDF")
        pdf_docs = st.file_uploader(" upload Files",type="pdf",accept_multiple_files=True)
        bt = st.button("Extraction")
        # ok = False
        if bt:
            with st.spinner("Processing ...."):
                # Get chunks
                chunks = get_extract_chunks(pdf_docs)
                # ok = True
                ## create vector store
                vectorstore = create_vectorstore(chunks)
                st.session_state.pdf_processed = True
                st.session_state.conversation = get_conversation_chain(vectorstore)
       
    # if ok:
    #     for chunk in chunks:
    #         st.success(chunk)
       
 
   
   
   
 
 
if __name__ == '__main__':
    main()
 
 
 
 
 
 