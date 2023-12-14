from langchain import HuggingFacePipeline, LlamaCpp
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text



def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def handle_userinput(user_question):
    docs = st.session_state.vector_store.similarity_search(query=user_question, k=3)
    answer = st.session_state.qa_chain.run(
        input_documents=docs, question=user_question)
    
    st.write(user_template.replace(
        "{{MSG}}", user_question), unsafe_allow_html=True)
    st.write(bot_template.replace(
        "{{MSG}}", answer), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Question Answering with PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    st.header("Question Answering with PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)

                vectorstore = get_vectorstore(text_chunks)
                st.session_state.vector_stor=vectorstore
                st.session_state.pdf_text = raw_text

                # Load model and create QA chain
                llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
                st.session_state.qa_chain = load_qa_chain(llm=llm,chain_type="map_reduce")


if __name__ == '__main__':
    main()
