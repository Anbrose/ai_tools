import requests
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Using pretrained classification to distinguish AI and human")

    parser.add_argument('--hf_token',
                        type=str,
                        default=None,
                        help=None)

    args = parser.parse_args()

    return args


args = parse_args()
API_URL = "https://api-inference.huggingface.co/models/nicky007/stable-diffusion-logo-fine-tuned"
headers = {"Authorization": "Bearer {}".format(args.hf_token)}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content


image_bytes = query({
    "inputs": "Astronaut riding a horse",
})

# You can access the image with PIL.Image for example
import io
from PIL import Image
image = Image.open(io.BytesIO(image_bytes))
image.save("./results/generated_logo.png")