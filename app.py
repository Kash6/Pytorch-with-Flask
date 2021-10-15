import io
import json
import torch
import numpy as np
import cv2
from utils.transforms import get_no_aug_transform
from models.generator import Generator
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from torchvision import transforms
import subprocess
import tempfile
import re
import time

app = Flask(__name__)
pretrained_dir = "C:/Users/akash/Downloads/test/_trained_netG65.pth"
netG = Generator().to(torch.device('cpu'))
netG.eval()
netG.load_state_dict(torch.load(pretrained_dir, map_location=torch.device('cpu')))

def inv_normalize(img):
    # Adding 0.1 to all normalization values since the model is trained (erroneously) without correct de-normalization
    mean = torch.Tensor([0.485, 0.456, 0.406]).to(torch.device('cpu'))
    std = torch.Tensor([0.229, 0.224, 0.225]).to(torch.device('cpu'))

    img = img * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
    img = img.clamp(0, 1)
    return img

def predict_images(image_list):
    trf = get_no_aug_transform()
    image_list = torch.from_numpy(np.array([trf(img).numpy() for img in image_list])).to(torch.device('cpu'))
    print("lolz2")
    with torch.no_grad():
        generated_images = netG(image_list)
    generated_images = inv_normalize(generated_images)
    print("LOLZ3")
    pil_images = []
    for i in range(generated_images.size()[0]):
        generated_image = generated_images[i].cpu()
        pil_images.append(TF.to_pil_image(generated_image))
    return pil_images 

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = Image.open(file).convert('RGB')
        print("lolz")
        out_img = predict_images([img_bytes])[0]
        print("lolz4")
        return out_img


if __name__ == '__main__':
    app.run()