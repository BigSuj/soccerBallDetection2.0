from flask import Flask, request, render_template
import numpy as np
import onnxruntime as ort
from azure.storage.blob import BlobServiceClient
import tempfile
from PIL import Image
import azure_blob as ab



app = Flask(__name__)

# Load the trained model and class names
def load_model_from_azure_blob_storage(connection_string, container_name, blob_name):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)

    with tempfile.NamedTemporaryFile() as tmp:
        with open(tmp.name, "wb") as model_file:
            model_file.write(blob_client.download_blob().readall())
        sess = ort.InferenceSession(tmp.name)

    return sess

connection_string = 'DefaultEndpointsProtocol=https;AccountName=sujmlmodels;AccountKey=GzBwpiN+3m2nMiKdMrMJrZ8E3g+S+H1c/EkkUuEO4Dm+rgWdsuUhjedntgAl7XgWyuTaF8VNozHa+AStfIKtmg==;EndpointSuffix=core.windows.net'
container_name = 'moodels'
blob_name = 'model.onnx'

sess = load_model_from_azure_blob_storage(connection_string, container_name, blob_name)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
class_names = ['Not a soccer ball', 'soccer ball']  # list of class names

# Define a function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the input shape of the model
    image = Image.open(image)
    image = image.resize((224, 224))
    image = image.convert('RGB')
    
    # Convert the image to a NumPy array and scale the pixel values to [0, 1]
    x = np.array(image, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    
    return x

@app.route('/', methods=['GET', 'POST'])
def predict_image():
    if request.method == 'POST':
        # Get the uploaded file from the form
        global file
        file = request.files['image']
        
        # Load the image and preprocess it
        x = preprocess_image(file)
        
        # Make a prediction using the model
        y = sess.run([output_name], {input_name: x})
        class_idx = np.argmax(y)
        class_label = class_names[class_idx]
        
        # Render the results page with the predicted class label
        return render_template('results.html', class_label=class_label)
    else:
        # Render the form page to upload an image
        return render_template('form.html')

@app.route('/answer', methods=['POST'])
def answer():
    # Get the predicted label and the actual label from the form data
    predicted_label = request.form['predicted_label']
    actual_label = request.form['actual_label']

    label = 0
    if actual_label == 'Soccer Ball':
        label = 1
    ab.upload_image(file, label)

    # Render the results page with the predicted label and the "Thank you" message
    return render_template('results.html', class_label=predicted_label, submitted=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
