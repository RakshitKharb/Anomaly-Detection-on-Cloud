import os
import pandas as pd
from azure.storage.blob import BlobClient
from io import StringIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

STORAGE_ACCOUNT = os.getenv("STORAGE_ACCOUNT")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")
BLOB_NAME = "cost-analysis.csv"  # Replace with your actual CSV file name
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

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
    print("First 5 rows of the dataset:")
    print(df.head())
except Exception as e:
    print(f"Failed to read CSV data: {e}")
    exit(1)
