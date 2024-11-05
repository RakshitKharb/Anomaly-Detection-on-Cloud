import os
import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
import joblib
from dotenv import load_dotenv
from azure.ai.ml.entities import Model  # Import the Model class

# Load environment variables
load_dotenv()

STORAGE_ACCOUNT = os.getenv("STORAGE_ACCOUNT")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")
BLOB_NAME = "cost-analysis.csv"  # Replace with your actual CSV file name
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
TARGET_COLUMN = "Cost"  # Set to your actual target column name
SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME")
RESOURCE_GROUP = os.getenv("RESOURCE_GROUP")

# Initialize MLClient
try:
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME
    )
    print(f"Connected to workspace: {WORKSPACE_NAME}")
except Exception as e:
    print(f"Error initializing MLClient: {e}")
    exit(1)

# Access Blob Storage
try:
    blob = BlobClient.from_connection_string(conn_str=CONNECTION_STRING, container_name=CONTAINER_NAME, blob_name=BLOB_NAME)
    downloader = blob.download_blob()
    data = downloader.content_as_text()
    print(f"Successfully downloaded blob: {BLOB_NAME}")
except Exception as e:
    print(f"Failed to download blob: {e}")
    exit(1)

# Load CSV into pandas DataFrame
try:
    df = pd.read_csv(StringIO(data))
    print("Initial DataFrame:")
    print(df.head())
except Exception as e:
    print(f"Failed to read CSV data: {e}")
    exit(1)

# Data Cleaning
df_cleaned = df.drop_duplicates()

# Convert 'UsageDate' to datetime
df_cleaned['UsageDate'] = pd.to_datetime(df_cleaned['UsageDate'], errors='coerce')

# Handle any potential conversion errors
if df_cleaned['UsageDate'].isnull().any():
    print("Warning: Some 'UsageDate' entries could not be converted and are set as NaT.")

# Extract additional features from 'UsageDate'
df_cleaned['Year'] = df_cleaned['UsageDate'].dt.year
df_cleaned['Month'] = df_cleaned['UsageDate'].dt.month
df_cleaned['Day'] = df_cleaned['UsageDate'].dt.day

# Drop the original 'UsageDate' if not needed
df_cleaned = df_cleaned.drop('UsageDate', axis=1)

# Feature Selection
X = df_cleaned.drop(TARGET_COLUMN, axis=1)
y = df_cleaned[TARGET_COLUMN]

# Encode categorical variables in X
X = pd.get_dummies(X, drop_first=True)

# Optional: Handle categorical target variable if needed
# Since 'Cost' is numerical, no encoding is necessary

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training completed.")

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)  # Updated to use the new function
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")

# Save the Model
model_filename = "model.pkl"
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")

# Create a Model entity
model_entity = Model(
    name="cost-prediction-model",
    path=model_filename,
    description="Random Forest Regressor for Cost Prediction",
    tags={"type": "regression", "algorithm": "RandomForest"}
)

# Upload the Model to Azure ML Workspace
try:
    ml_client.models.create_or_update(model_entity)
    print("Model registered in Azure ML Workspace.")
except Exception as e:
    print(f"Failed to register model: {e}")
    exit(1)
