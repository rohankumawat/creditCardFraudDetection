from flask import Flask, request, jsonify, render_template
import joblib
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

# Load environment variables from .env
load_dotenv()

# Set your OpenAI API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Load the trained model
model = joblib.load('models/LogisticRegression_model.pkl')
# Load the trained scaler
scaler = joblib.load('models/scaler.pkl')

# Initialize the OpenAI language model
llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

# Define the function for generating explanations using LangChain
def generate_explanation(features, prediction):
    """
    Generate a natural language explanation for the model's prediction using LangChain.
    """
    # Define the system message and prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer all questions to the best of your ability.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # Create the template for features and prediction
    template = (
        "The transaction has the following features: {features}. "
        "The model predicted {prediction}. Explain why the model made this prediction."
    )

    # Format the template with actual features and prediction
    message_content = template.format(features=features, prediction=prediction)

    # Invoke the model to generate the explanation
    response = llm.invoke([HumanMessage(content=message_content)])

    # Return the content of the response
    return response.content

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get features from the form
    input_data = request.form['features']  # features except Time and Amount
    time = float(request.form['time'])
    amount = float(request.form['amount'])

    # Convert the features into a list of floats
    features = [float(x) for x in input_data.split(',')]

    # Scale Time and Amount separately
    scaled_time = scaler.transform([[time]])[0][0]  # Scaling Time
    scaled_amount = scaler.transform([[amount]])[0][0]  # Scaling Amount

    # Append the scaled Time and Amount to the features
    features.append(scaled_time)
    features.append(scaled_amount)

    # Make prediction using the trained model
    prediction = model.predict([features])
    prediction_text = 'Fraud' if prediction == 1 else 'Not Fraud'

    # Return result
    # result = 'Fraud' if prediction[0] == 1 else 'Not Fraud'

    # Generate explanation using OpenAI
    explanation = generate_explanation(features, prediction_text)

    return render_template('index.html', prediction_text=f'Prediction: {prediction_text}', explanation_text=f'Explanation: {explanation}')


if __name__ == "__main__":
    app.run(debug=True)
