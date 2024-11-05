import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

# List all registered models
try:
    models = ml_client.models.list()
    print("Registered Models:")
    for model in models:
        print(f"Name: {model.name}, Version: {model.version}, Description: {model.description}")
except Exception as e:
    print(f"Failed to list models: {e}")
    exit(1)
