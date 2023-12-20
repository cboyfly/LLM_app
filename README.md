# Insurance Assistant

Insurance Assistant is a langchain RetrievalQA assistant designed to answer basic questions about insurance benefits forms. Utilizing OpenAI’s “gpt-3.5-turbo-instruct” as the base model, this assistant provides a user-friendly way to interpret and understand insurance documents.

## Description

Insurance Agent allows users to upload their insurance benefits coverage form and ask various questions to gain clarity on their insurance plan's specifics. The project leverages advanced language model technology to deliver accurate and contextually relevant answers, making it easier for users to navigate the often confusing world of insurance paperwork.

## Installation

To set up the Insurance Agent on your local machine, follow these steps:

1. Clone the repository to your local machine.
2. Install the required packages:
pip install langchain pandas python-dotenv streamlit textwrap

## Running the Application

- **To access the assistant**:

This command will start the Streamlit application and provide an interface to interact with the insurance assistant.

streamlit run main.py




- **For prompt evaluation**:
Giskard uses LLM-assisted detectors for evaluating the effectiveness and accuracy of the prompts.

python giskard_eval.py


Insurance Agent is currently in development. 
