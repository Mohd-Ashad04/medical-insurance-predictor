from flask import Flask, render_template, request
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained pipeline from pickle
with open('model.pkl', 'rb') as f:
    pipeline = pickle.load(f)

@app.route('/')
def home():
    """Render the input form page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Process form data, run prediction, and render result page."""
    # Collect form inputs
    input_data = {
        'age': int(request.form['age']),
        'sex': request.form['sex'],
        'bmi': float(request.form['bmi']),
        'children': int(request.form['children']),
        'smoker': request.form['smoker'],
        'region': request.form['region'],
        'exercise_level': request.form['exercise_level'],
        'alcohol_consumption': request.form['alcohol_consumption'],
        'chronic_disease': int(request.form['chronic_disease']),
        'family_history': request.form['family_history'],
        'married': request.form['married'],
        'occupation_type': request.form['occupation_type'],
    }

    # Feature engineering (must match training)
    input_data['bmi_smoker'] = input_data['bmi'] * (1 if input_data['smoker'] == 'yes' else 0)
    input_data['age_chronic'] = input_data['age'] * input_data['chronic_disease']
    input_data['bmi_age'] = input_data['bmi'] * input_data['age']
    input_data['children_smoker'] = input_data['children'] * (1 if input_data['smoker'] == 'yes' else 0)

    # Convert to DataFrame
    df_input = pd.DataFrame([input_data])

    # Predict using the loaded pipeline
    prediction_value = pipeline.predict(df_input)[0]
    prediction = round(prediction_value, 2)

    # Render result page
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
