import os
import yaml
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, Model, OnlineEndpoint, OnlineDeployment
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("RESOURCE_GROUP")
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME")

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

# Define and register Environment
env = Environment(
    name="cost-prediction-env",
    description="Environment for Cost Prediction Model",
    conda_file="environment.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"  # Ensure this image is appropriate
)

try:
    registered_env = ml_client.environments.create_or_update(env)
    print("Environment 'cost-prediction-env' registered.")
except Exception as e:
    print(f"Failed to register environment: {e}")
    exit(1)

# Define Online Endpoint
endpoint_name = "cost-prediction-endpoint"

# Check if Endpoint Exists; if not, create it
try:
    endpoint = ml_client.online_endpoints.get(name=endpoint_name)
    print(f"Endpoint '{endpoint_name}' already exists.")
except Exception:
    try:
        endpoint = OnlineEndpoint(
            name=endpoint_name,
            description="Endpoint for Cost Prediction Model",
            auth_mode="key"  # Options: 'key' or 'aad'
        )
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        print(f"Endpoint '{endpoint_name}' created.")
    except Exception as e:
        print(f"Failed to create endpoint: {e}")
        exit(1)

# Retrieve the registered model
try:
    model = ml_client.models.get(name="cost-prediction-model", version="1")
    print(f"Retrieved model: {model.name}, version: {model.version}")
except Exception as e:
    print(f"Failed to retrieve model: {e}")
    exit(1)

# Define Deployment Configuration
deployment_name = "cost-prediction-deployment"

# Create Deployment Object using MLClient's Deployment methods
try:
    deployment = OnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        environment=registered_env,
        instance_type="Standard_F4s",  # Choose an appropriate instance type
        scale_settings={
            "min_replicas": 1,
            "max_replicas": 1,
            "scale_type": "default"
        }
    )
    # Use begin_create_or_update instead of create_or_update
    ml_client.online_deployments.begin_create_or_update(deployment).result()
    print(f"Deployment '{deployment_name}' created.")
except Exception as e:
    print(f"Failed to create deployment: {e}")
    exit(1)

# Set the Deployment as Default
try:
    ml_client.online_endpoints.begin_set_default(
        name=endpoint_name,
        deployment_name=deployment_name
    ).result()
    print(f"Deployment '{deployment_name}' set as default for endpoint '{endpoint_name}'.")
except Exception as e:
    print(f"Failed to set default deployment: {e}")
    exit(1)

# Get Endpoint Details
try:
    endpoint = ml_client.online_endpoints.get(name=endpoint_name)
    print(f"Endpoint URL: {endpoint.scoring_uri}")
    print(f"Primary Key: {endpoint.keys[0]}")
except Exception as e:
    print(f"Failed to retrieve endpoint details: {e}")
    exit(1)
