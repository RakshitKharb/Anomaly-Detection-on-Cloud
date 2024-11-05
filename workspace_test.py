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
    print(f"Connected to workspace: {WORKSPACE_NAME} in resource group: {RESOURCE_GROUP}")
except Exception as e:
    print(f"Failed to connect to workspace: {e}")
