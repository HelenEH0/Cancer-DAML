import gradio as gr
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

importantFeatures = ['Air Pollution', 'Alcohol use', 'Obesity', 'Passive Smoker',
                     'Coughing of Blood', 'Fatigue', 'Shortness of Breath', 'Dry Cough']


def make_prediction(csv_file):
    file = csv_file.name

    # Read the uploaded CSV file
    data = pd.read_csv(file)

    with open("filename.pkl", "rb") as f:
        clf = pickle.load(f)
    predictions = []
    for index, row in data.iterrows():
        input_data = row[importantFeatures]
        pred = clf.predict([input_data])[0]
        predictions.append(list(input_data) + [pred])
    predictions_df = pd.DataFrame(
        predictions, columns=importantFeatures + ['prediction'])

    # Rename the column name to "level of risk"
    predictions_df = predictions_df.rename(
        columns={'prediction': 'level of risk'})

    # Map the prediction values to "Low" and "High"
    predictions_df['level of risk'] = predictions_df['level of risk'].map(
        {1: 'Low', 2: 'Low', 3: 'Low', 4: 'Low', 5: 'High', 6: 'High', 7: 'High', 8: 'High'})

    # Convert predictions_df to JSON format
    predictions_json = predictions_df.to_json(orient='records')

    print(predictions_json)


# Return predictions as CSV string

    return predictions_df.to_json(orient='records')


# Create the input component for Gradio to accept a file upload
input_csv = gr.inputs.File(label="Upload CSV file")

# Create the output component for Gradio to output text
outputs = gr.outputs.JSON()

# Define the Gradio interface
iface = gr.Interface(fn=make_prediction, inputs=input_csv,
                     outputs=outputs, title="Cancer Predictor")

# Launch the interface
iface.launch(share=True)
