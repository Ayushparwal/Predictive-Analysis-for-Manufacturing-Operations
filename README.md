# Predictive-Analysis-for-Manufacturing-Operations
In this repo I have used fastAPI, to make a work flow and how to upload the dataset and train the model then make predicitons.

How to run this code I will tell you.
This project provides a RESTful API to predict machine downtime or production defects using a manufacturing dataset. The API is built using FastAPI and includes endpoints to upload data, train a machine learning model, and make predictions.

## Features
- Upload a manufacturing dataset (CSV format).
- Train a logistic regression model on the uploaded dataset. depends what you have to use as model I have used RandomForestClassifier.
- Predict downtime based on temperature and runtime.

### Prerequisites
- Python 3.8 or higher
- Required Python libraries:
  - fastapi
  - uvicorn
  - pandas
  - scikit-learn
  - joblib

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/Ayushparwal/Predictive-Analysis-for-Manufacturing-Operations.git
   cd Predictive-Analysis-for-Manufacturing-Operations
   ```

2. Install the required libraries:
   ```sh
   pip install -r requirements.txt
   ```

3. Run the application:
   ```sh
   uvicorn app:app --reload
   ```

Go to this
4. Access the API at: 
  ```sh
  http://127.0.0.1:8000/docs
  ```

ps: make sure to upload the dataset of trail_data.csv for more read pdf in the repo.
