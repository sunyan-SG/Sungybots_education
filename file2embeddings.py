import filetype  # For file type detection
from docx import Document  # For Word file processing
from PyPDF2 import PdfReader  # For PDF file processing
import openpyxl

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st


def get_file_text(uploaded_files):
    text = ""
    for file in uploaded_files:
        kind = filetype.guess(file)  # Detect file type
        if kind is None:
            print(f'Cannot determine file type for: {file.name}')
            continue

        if kind.mime == 'application/pdf':
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif kind.mime == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            document = Document(file)
            for paragraph in document.paragraphs:
                text += paragraph.text
        elif kind.mime == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            workbook = openpyxl.load_workbook(file)
            sheet = workbook.active  # Assuming you want the active sheet
            for row in sheet.iter_rows():
                row_text = ""  # Collect text for each row
                for cell in row:
                    if cell.value is not None:
                        row_text += str(cell.value) + "\n"  # Adjust concatenation as needed
                text += row_text + "\n"  # Add row to overall text
        else:
            print(f'Unsupported file type: {file.name}')

    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'})
    #embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


st.set_page_config(page_title="file uploader",
                   layout="wide",
                   page_icon=":books::parrot:")

st.title("File Uploader for Sungy Customers")
st.session_state.pdf_docs = st.file_uploader(
    "Upload your PDFs here and click on 'Process'", type={"pdf", "docx", ".xlsx"}, accept_multiple_files=True)
if st.button("Process"):
    # with st.spinner("Processing"):
    # get pdf text
    raw_text = get_file_text(st.session_state.pdf_docs)
    st.write(raw_text)
    # get the text chunks
    text_chunks = get_text_chunks(raw_text)
    # create vector store
    vectorDB = get_vectorstore(text_chunks)

    # Save faiss vector to disk or store in a database/cloud storage
    file = st.session_state.pdf_docs[0]
    #st.write(file)
    #name = 'vectorDB_'+file.name
    name = 'vectorDB'
    vectorDB.save_local(name)

    st.success('PDF uploaded and embeddings created successfully!', icon="✅")