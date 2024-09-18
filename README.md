# Credit Card Fraud Detection System with MLOps Integration

---

## **Project Overview**

This project is focused on building a Credit Card Fraud Detection System that utilizes machine learning techniques to detect fraudulent transactions. The project will incorporate MLOps practices to ensure the entire lifecycle of the machine learning model is automated and scalable. The project also includes a real-time dashboard for visualizing fraud detection insights.
Features

- **Data Ingestion & Preprocessing**: Automated data pipeline for ingesting and cleaning transaction data.
- **Fraud Detection Models**: Supervised (Logistic Regression, Decision Trees) and Unsupervised (Clustering) models to detect fraud.
- **Model Deployment**: Dockerized machine learning models with continuous integration and deployment (CI/CD).
- **Model Monitoring**: Real-time performance monitoring using Prometheus and Grafana.
- **Dashboard**: Interactive dashboard for real-time visualization of results.

## **Tech Stack**
- **Backend**: Flask (Python)
- **Machine Learning Model**: Logistic Regression, Random Forest, XGBoost
- **Frontend**: HTML, CSS
- **Data Preprocessing**: Pandas, Scikit-learn (Standard Scaler for feature scaling)
- **Deployment**: Docker (For easy containerization)
- **Future Deployment Plans**: Google Cloud, Vercel, or Azure for hosting

---

## **How It Works**

The application allows users to:
- **Select a Machine Learning Model**: Choose between Logistic Regression, Random Forest, and XGBoost.
- **Enter Transaction Features**: Users input transaction details (features, time, and amount).
- **Get Fraud Predictions**: The app predicts whether the transaction is fraudulent or not.
- **Receive Explanation**: Uses **LangChain/OpenAI** API to generate a natural language explanation of the prediction.
  
### **Features**
- **Risk Segmentation**: Allows users to classify transactions as high-risk, medium-risk, or low-risk based on model predictions.
- **Fraud Probability**: Displays the likelihood of a transaction being fraudulent as a percentage, helping users gauge model confidence.
- **Graphs and Visualizations** (Future Work): Planned visualizations to help users understand patterns in transaction risks.
  
---

## **Getting Started**

### **Prerequisites**
- Python 3.11 (or higher)
- Docker (for containerization and deployment)
- Optional: Google Cloud or other cloud platforms if deploying the app

### **1. Clone the Repository**

```bash
git clone https://github.com/yourusername/creditCardFraudDetection.git
cd creditCardFraudDetection
```

### **2. Set Up the Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # For macOS/Linux
# OR
venv\Scripts\activate  # For Windows
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Set Up the Environment Variables**

Create a `.env` file and add your OpenAI API key (for explanation generation via LangChain).

```bash
OPENAI_API_KEY=your-api-key
```

### **5. Run the Flask App**

You can now start the Flask server locally.

```bash
flask run
```

The app will be accessible at `https://127.0.0.1:5000`.

### **6. Build and Run with Docker (Optional)**

To build with Docker image:

```bash
docker build -t my-flask-app .
```

To run with Docker container:

```bash
docker run -d -p 5000:5000 my-flask-app
```

### **7. Make Predictions**

Navigate to the app in your browser and enter transaction details:

- **Select a model**: Choose from Logistic Regression, Random Forest, or XG Boost.
- **Enter Transaction Features**: Input transaction features (28 features), time, and amount.

**Testing via cURL**

You can also make predictions using `cURL`:

```bash
curl -X POST http://127.0.0.1:5000/predict \
    -d "features=1.0,-1.2,0.5,2.3,-0.7,0.1,-0.4,1.5,0.9,0.2,0.8,-0.6,0.3,-0.2,-1.0,0.5,0.1,-0.7,0.2,1.1,-0.3,-0.8,1.4,0.6,0.4,-1.3,0.1,4.6" \
    -d "time=12000" \
    -d "amount=100.0" \
    -d "model=Logistic Regression"
```

---

## Future Work

- [ ] **Enhanced Visualization**: Incorporate Plotly graphs and charts.
- [ ] **Interactive Transaction Exploration**: Allow users to upload transaction datasets and receive a batch analysis of potential fraud cases.
- [ ] **Real-Time Fraud Detection**: Integrate with APIs to perform real-time fraud detection on live transactions. (Or perform Simulation)
- [ ] **Model Tuning**: Hyperparameter Tuning and Model Evaluation
- [ ] **Deploy on Cloud**

---