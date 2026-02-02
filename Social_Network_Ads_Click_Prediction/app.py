import gradio as gr
import pandas as pd
import pickle
import numpy as np

with open('svm_model.pkl', 'rb') as f:
    svm_model_pipeline = pickle.load(f)

def predict_purchase(Gender, Age, EstimatedSalary):
    input_df = pd.DataFrame([[
        Gender, Age, EstimatedSalary
    ]],
    columns = [
        'Gender', 'Age', 'EstimatedSalary'
    ])

    prediction = svm_model_pipeline.predict(input_df)[0]

    return "Purchased" if prediction == 1 else "Not Purchased"

inputs = [
    gr.Radio(["Male", "Female"], label="Gender"),
    gr.Number(label="Age"),
    gr.Number(label="Estimated Salary")
]

app = gr.Interface(
    fn = predict_purchase,
    inputs = inputs,
    outputs = "text",
    title = "Social Network Ads Purchase Prediction"
)

app.launch(share=True)