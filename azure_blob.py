from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from dotenv import load_dotenv
import os

load_dotenv()

connection_string = os.getenv("CONNECTION_STRING")
ball_name = 'ball'
no_ball_name = 'no-ball'


def upload_image(file, label):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        ball_client = blob_service_client.get_container_client(ball_name)
        no_ball_client = blob_service_client.get_container_client(no_ball_name)
    except Exception as e:
        print('Could not connect to azure blob storage.')
        print(e)
        return
    
    container_client = ball_client
    if label == 0:
        container_client = no_ball_client
    
    filename = file.filename
    blob_client = container_client.get_blob_client(blob=filename)

    file_contents = file.read()
    blob_client.upload_blob(file_contents, overwrite=True)

