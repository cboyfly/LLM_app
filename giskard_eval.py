# Packages for the model wrapping and validation

import numpy as np
import pandas as pd
import giskard
import langchain_helper as lch


# Make a pandas dataframe of sample input prompts

sample_questions = [
    "What is the annual deductible for individual coverage under this plan?",
    "Does this insurance plan cover dental services, and if so, what is the extent of the coverage?",
    "Are there any specific exclusions or limitations for pre-existing conditions in this insurance policy?",
    "What is the maximum out-of-pocket expense for a family enrolled in this plan?",
    "How does the plan handle emergency medical services, especially when out of network?"
    "What is the derivative of sin(x) with respect to x?"
]

raw_data = pd.DataFrame(data={"Questions": sample_questions})
giskard_dataset = giskard.Dataset(raw_data, target=None)

def prediction_function(df):
    """List of responses for each question in the dataframe"""
    db = lch.create_vector_db_from_pdf('https://hr.umich.edu/sites/default/files/uofm_cb_ppo_baag_2023.pdf')
    
    return [lch.get_response_from_query(db, data) for data in df["Questions"]]


giskard_model = giskard.Model(
    model=prediction_function,  # A prediction function that encapsulates all the data pre-processing steps and that could be executed with the dataset used by the scan.
    model_type="text_generation",  # Either regression, classification or text_generation.
    name="The LLM, which knows about benefits forms.",  # Optional.
    description="This model knows facts about insurance benefits coverage forms. This model responses strictly and shortly.",  # Is used to generate prompts during the scan.
    feature_names=["Questions"]  # Default: all columns of your dataset.
)

print(giskard_model.predict(giskard_dataset).prediction)

results = giskard.scan(giskard_model, giskard_dataset)

giskard.display(results)