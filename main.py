import streamlit as st
import langchain_helper as lch
import textwrap

st.title('Insurance Assistant')

with st.sidebar:
    with st.form(key = 'my_form'):
        pdf_url = st.sidebar.text_area(
            label = "What is the pdf URL?",
            max_chars = 75
        )
        df = st.sidebar.text_area(
            label = "Provide a pandas dataframe of questions you would like answered about your video",
            max_chars = 50,
            key = 'df'
        )

        submit_button = st.form_submit_button(label = 'Submit')

if df and pdf_url:
    vector_db = lch.create_vector_db_from_pdf
    response, docs = lch.get_response_from_query(vector_db, df)
    st.subheader("Answer")
    st.text(textwrap.fill(response, width = 80))