import pickle
import numpy as np
import pandas as pd
import torch
import tkinter as tk 
import tensorflow as tf
from utils.vae import VAE
from sklearn.preprocessing import MinMaxScaler
from py.gui import gui
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50
from PIL import Image, ImageTk
from tkinter import scrolledtext

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('models/sentiment_model.pkl', 'rb') as file:
    sentiment_model = pickle.load(file)

resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

vae = VAE().to(device)
vae.load_state_dict(torch.load('models/vae_model.pth', map_location=device), strict=False)
vae.eval()

df = pd.read_csv("data/shapes.csv")
emotions = df.columns[7:]

def scoreToLatent(match_score, latent_dim=128):
    scaler = MinMaxScaler()
    normalize = scaler.fit_transform(np.array([[match_score]]))
    normalize_scalar = normalize.item() if normalize.ndim > 0 else normalize
    vector = torch.randn(1, latent_dim).float().to(device) * float(normalize_scalar)
    return vector

def latentToImage(vector):
    vector = vector.float()
    vector = vector.view(1, -1)
    generate = vae.decoder(vector)
    generate = generate.squeeze(0).detach().cpu().numpy()
    generate = generate.transpose(1, 2, 0)
    generate = np.clip(generate, 0, 1)
    generate = (generate * 255).astype(np.uint8)
    return Image.fromarray(generate)

def analyzeSentiment(input):
    sentiment_scores = sentiment_model.predict([input])[0]
    return dict(zip(emotions, sentiment_scores))

def calculateMatch(dataset, sentiment_scores):
    for emotion in emotions:
        dataset[emotion] = dataset[emotion] * sentiment_scores.get(emotion, 0)
    dataset['match_score'] = dataset[emotions].sum(axis=1)
    return dataset

def inputChange(input_message, history, input_text):
    sentiment_scores = analyzeSentiment(input_message)
    image_scores = calculateMatch(df.copy(), sentiment_scores)
    top_matches = image_scores.sort_values(by="match_score", ascending=False).head(1)

    match_score = top_matches['match_score'].iloc[0]
    vector = scoreToLatent(match_score)
    generate = latentToImage(vector)

    img_tk = ImageTk.PhotoImage(generate)
    history.image_create(tk.END, image=img_tk)
    history.insert(tk.END, "\n")

    history.yview(tk.END)
    input_text.delete(0, tk.END)
    history.image = img_tk

gui(inputChange)



