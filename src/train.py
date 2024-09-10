import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.metrics import RocCurveDisplay
from imblearn.over_sampling import SMOTE
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from data_ingestion import load_and_preprocess

# Dictionary of models to train
models = {
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    # Add more models here if needed
}

# Function to select cross-validation method
def get_cross_validation_method(cv_type, n_splits=5):
    if cv_type == 'KFold':
        return KFold(n_splits=n_splits, random_state=42, shuffle=True)
    elif cv_type == 'StratifiedKFold':
        return StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
    elif cv_type == 'LeaveOneOut':
        return LeaveOneOut()
    else:
        raise ValueError("Invalid cross-validation type. Choose from 'KFold', 'StratifiedKFold', 'LeaveOneOut'.")

def plot_and_log_roc_curve(model, X_test, y_test, model_name):
    """
    Plot and log the ROC curve for a given model.
    """
    # Predict probabilities for the positive class (class=1)
    y_probs = model.predict_proba(X_test)[:, 1]

    # Compute ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
    plt.title(f"ROC Curve for {model_name} (AUC = {roc_auc:.2f})")

    # Save the plot
    plt_path = f"models/{model_name}_roc_curve.png"
    plt.savefig(plt_path)
    plt.close()

    # Log the ROC curve plot to MLflow
    mlflow.log_artifact(plt_path)
    print(f"ROC Curve for {model_name} logged to MLflow.")

def train_and_evaluate_model(model_name, model, X_train, y_train, X_test, y_test, cv_method):
    """
    Train and evaluate a given model using cross-validation, plot the ROC curve, and log the results to MLflow.
    """
    with mlflow.start_run(run_name=model_name):
        # Apply cross-validation
        scores = cross_val_score(model, X_train, y_train, cv=cv_method, scoring='accuracy')
        avg_score = scores.mean()

        # Log parameters, metrics, and model to MLflow
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("cv_type", str(cv_method))
        mlflow.log_metric("avg_cv_accuracy", avg_score)

        # Train on the full training set and log the model
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, f"{model_name}_model")
        
        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        clf_report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", clf_report['1']['precision'])
        mlflow.log_metric("recall", clf_report['1']['recall'])
        mlflow.log_metric("f1_score", clf_report['1']['f1-score'])

        # Plot and log the ROC curve
        plot_and_log_roc_curve(model, X_test, y_test, model_name)
        
        # Save the model locally
        joblib.dump(model, f"models/{model_name}_model.pkl")
        print(f"{model_name} model saved to 'models/{model_name}_model.pkl' with avg accuracy: {avg_score:.4f}")

def main(cv_type='KFold', n_splits=5):
    # Load and preprocess the data
    data = load_and_preprocess('data/raw/creditcard.csv')
    
    # Split the data into features (X) and target (y)
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE to balance the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Get cross-validation method
    cv_method = get_cross_validation_method(cv_type, n_splits)

    # Train and evaluate each model
    for model_name, model in models.items():
        train_and_evaluate_model(model_name, model, X_train_resampled, y_train_resampled, X_test, y_test, cv_method)

if __name__ == "__main__":
    main(cv_type='StratifiedKFold', n_splits=5)  # Change this as needed
