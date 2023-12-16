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
        query = st.sidebar.text_area(
            label = "Ask about the insurance form?",
            max_chars = 75,
            key = 'query'
        )
        # context = st.sidebar.text_area(
        #     label = "Provide basic info about the form being parsed:",
        #     max_chars = 125,
        #     key = 'context'
        # )

        submit_button = st.form_submit_button(label = 'Submit')

if query and pdf_url:
    db = lch.create_vector_db_from_pdf(pdf_url)
    response = lch.get_response_from_query(db, query)
    st.subheader("Answer: ")
    st.text(textwrap.fill(response, width = 80))