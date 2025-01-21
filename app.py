from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime

app = FastAPI()

DATA_DIR = "data"
MODEL_DIR = "models"
dataset = None
model = None

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
   
    global dataset
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    dataset = pd.read_csv(file_path)
    required_columns = {"Machine_ID", "Temperature", "Run_Time", "Downtime_Flag"}
    if not required_columns.issubset(dataset.columns):
        raise HTTPException(status_code=400, detail=f"Dataset must include columns: {required_columns}")
    
    return {"message": "Dataset uploaded successfully!", "file_path": file_path}


@app.post("/train")
async def train_model():
    
    global dataset, model
    if dataset is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded.")

    X = dataset[["Temperature", "Run_Time"]]
    y = dataset["Downtime_Flag"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    model_filename = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)
    joblib.dump(model, model_path)

    return {
        "message": "Model trained successfully!",
        "accuracy": accuracy,
        "f1_score": f1,
        "model_path": model_path
    }


@app.post("/predict")
async def predict(data: dict):
    
    global model
    if model is None:
        raise HTTPException(status_code=400, detail="Model not trained.")

    if not {"Temperature", "Run_Time"}.issubset(data.keys()):
        raise HTTPException(status_code=400, detail="Input data must include 'Temperature' and 'Run_Time'.")

    input_data = pd.DataFrame([data])

    prediction = model.predict(input_data)[0]
    confidence = max(model.predict_proba(input_data)[0])

    downtime = "Yes" if prediction == 1 else "No"
    return {"Downtime": downtime, "Confidence": round(confidence, 2)}


@app.get("/status")
async def status():
    
    return {"status": "API is running", "model_loaded": model is not None}
