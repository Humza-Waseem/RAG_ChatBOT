import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter


def  get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text



def get_text_chunks(raw_text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = splitter.split_text(raw_text)  # Use 'splitter', not 'text_splitter'
    return text_chunks



def main():
    load_dotenv()
    st.set_page_config(page_title = "Chat with countries", page_icon = "🌍", layout = "wide")

    st.header("chat with countries:")
    st.text_input("Ask Questions:")


    with st.sidebar:
        st.subheader("Countries")
        pdf_docs = st.file_uploader("Upload a file", accept_multiple_files=True)
        if st.button("proceed"):
            with st.spinner("loading..."):
                st.success("done")

                #get PDF text
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)
                


                #get text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)
                



if __name__ == '__main__':
  main()
