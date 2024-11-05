import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from azure.storage.blob import BlobClient
from io import StringIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

STORAGE_ACCOUNT = os.getenv("STORAGE_ACCOUNT")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")
BLOB_NAME = "cost-analysis.csv"  # Replace with your actual CSV file name
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
TARGET_COLUMN = "Cost"  # Set to your actual target column name

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

# Extract additional features from 'UsageDate' if useful
df_cleaned['Year'] = df_cleaned['UsageDate'].dt.year
df_cleaned['Month'] = df_cleaned['UsageDate'].dt.month
df_cleaned['Day'] = df_cleaned['UsageDate'].dt.day

# Drop the original 'UsageDate' if not needed
df_cleaned = df_cleaned.drop('UsageDate', axis=1)

# Feature Selection for EDA
numerical_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()
if TARGET_COLUMN in numerical_cols:
    numerical_cols.remove(TARGET_COLUMN)

# Create directory for plots
plots_dir = "plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Exploratory Data Analysis (EDA)

# 1. Distribution of numerical columns
for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.histplot(df_cleaned[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.savefig(f'{plots_dir}/distribution_{col}.png')
    plt.close()
    print(f"Saved distribution plot for {col}.")

# 2. Correlation Heatmap
plt.figure(figsize=(10, 8))
# Select only numeric columns for correlation
corr_matrix = df_cleaned.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig(f'{plots_dir}/correlation_heatmap.png')
plt.close()
print("Saved correlation heatmap.")

# 3. Boxplot of numerical features by target
for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=TARGET_COLUMN, y=col, data=df_cleaned)
    plt.title(f'Boxplot of {col} by {TARGET_COLUMN}')
    plt.savefig(f'{plots_dir}/boxplot_{col}_by_{TARGET_COLUMN}.png')
    plt.close()
    print(f"Saved boxplot for {col} by {TARGET_COLUMN}.")

print("EDA plots saved in the 'plots' directory.")
