import urllib.parse
import numpy as np
import torch
from PIL import Image

def preprocess_image(image_uri, target_size=(384, 384)):
    try:
        image = Image.open(urllib.parse.unquote(image_uri))
        image = image.resize(target_size, resample=Image.LANCZOS)
        image = np.array(image)
        image = image.transpose((2, 0, 1))
        return image
    except Exception as e:
        print("Error preprocessing image:", e)
        return None

def preprocess_images(image_uris):
    preprocessed_images = []
    for image_uri in image_uris:
        preprocessed_image = preprocess_image(image_uri)
        if preprocessed_image is not None:
            preprocessed_images.append(preprocessed_image)
    return np.array(preprocessed_images)

def process_output(tensor):
    output_array = tensor.detach().numpy()
    predicted_indices = np.argmax(output_array, axis=1)
    predicted_gbif_ids = [index for index in predicted_indices]
    return predicted_gbif_ids

def execute(image_uris):
    model = torch.jit.load('model.pt')
    processed_images = preprocess_images(image_uris)
    input_tensor = torch.tensor(processed_images, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
    return process_output(output)

if __name__ == "__main__":
    print(execute(["1.jpg", "2.jpg"]))