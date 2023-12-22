import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import joblib

app = Flask(__name__, template_folder='templates')

# Load the trained model and scaler
final_knn_regressor = joblib.load("final_knn_regressor.pkl")
scaler = joblib.load("scaler.pkl")

# Assuming selected columns from feature selection
selected_columns = ['High', 'Low', 'Open_Price', 'Adj_Close', 'Volume']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        input_data = request.form.to_dict()

        # Convert numerical values to float and remove commas
        input_data = {key: float(value.replace(',', '')) for key, value in input_data.items()}

        # Prepare input data as a DataFrame
        input_df = pd.DataFrame(data=input_data, index=[0])
        input_df = input_df[selected_columns]

        # Standardize the input data
        input_scaled = scaler.transform(input_df)

        # Make predictions
        prediction = final_knn_regressor.predict(input_scaled)

        # Debug statement to print input and prediction
        print("Input Data:", input_data)
        print("Scaled Input:", input_scaled)
        print("Prediction:", prediction)

        # Display the prediction on the result page
        return render_template('index.html', prediction=prediction[0])

    except Exception as e:
        # Print any exception for debugging
        print("Exception:", str(e))
        return render_template('index.html', prediction="Error occurred. Please check your input.")

if __name__ == '__main__':
    app.run(debug=True)
