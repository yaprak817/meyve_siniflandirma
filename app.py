# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 15:40:25 2025

@author: Lenovo
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# Model yükleme
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 5)  # Buraya sınıf sayını yaz (örneğin 5 sınıf)
model.load_state_dict(torch.load("model/classifier.pth", map_location=device))
model = model.to(device)
model.eval()

# Sınıf isimleri
class_names = ['Apple', 'Banana', 'Orange', 'Strawberry', 'Pineapple']  # Buraya kendi sınıflarını yaz

# Ön işleme
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Streamlit başlık
st.title(" Görüntü Sınıflandırıcı")
st.write("Bir görüntü yükleyin ve hangi sınıfa ait olduğunu görün!")

# Görüntü yükleme
uploaded_file = st.file_uploader("Bir Görüntü Yükle", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Yüklenen Görüntü', use_column_width=True)

    if st.button('Tahmin Et'):
        img = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img)
            _, preds = torch.max(outputs, 1)
            predicted_class = class_names[preds.item()]
        st.success(f"Bu görsel: **{predicted_class}** sınıfına ait!")  