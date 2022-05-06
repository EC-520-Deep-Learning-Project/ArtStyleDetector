import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
import json
import tensorflow as tf
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from keras.preprocessing.image import img_to_array
from scipy.special import softmax

TOPK = 5
scc_or_colab = 'colab'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Get the output into usable format
softmax_torch = nn.Softmax(dim = 1)

# Loading the models
@st.experimental_singleton
def load_vit(MODEL_PATH):

    our_ViT = timm.create_model('vit_huge_patch14_224_in21k', num_classes = 25)
    # to be added
    if scc_or_colab != None:
        our_ViT.load_state_dict(torch.load(MODEL_PATH))
    else:
        our_ViT.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device('cpu')))
    config = resolve_data_config({}, model=our_ViT)
    transform_vit = create_transform(**config)

    return our_ViT,transform_vit

@st.experimental_singleton
def load_effnet(MODEL_PATH):

    effnet_model = tf.keras.models.load_model(MODEL_PATH)

    return effnet_model

@st.cache
def get_locations():
    # Loactions for saved models
    if scc_or_colab == 'scc':
        location1 = '/projectnb/dl523/projects/Sarcasm/520 Project/best_ViT_one_layer.pth'
        location2 = '/projectnb/dl523/projects/Sarcasm/520 Project/effnetv2_model'
    elif scc_or_colab == 'colab':
        location1 = '/content/drive/Shareddrives/520 Project/Saved Models/ViT/best_ViT_one_layer.pth'
        location2 = '/content/drive/Shareddrives/520 Project/Saved Models/effnetv2_model'
    else:
        location1 = '/Users/colehunter/Desktop/520 Project/best_ViT_one_layer.pth'
        location2 = '/Users/colehunter/Desktop/520 Project/effnetv2_model'
    return location1,location2

@st.cache
def get_dictionary():
    with open('styles_dictionary.json') as f:
        data = f.read()

    return json.loads(data)

@st.cache
def eff_transform(img):
  eff_tensor = img.resize((223, 223))
  eff_tensor = img_to_array(eff_tensor)
  eff_tensor = eff_tensor*(1/255)
  eff_tensor = eff_tensor.reshape((1,eff_tensor.shape[0],eff_tensor.shape[1],eff_tensor.shape[2]))
  return eff_tensor

location1,location2 = get_locations()

# Load the trained vit
vit,vit_transform = load_vit(location1)
vit.to(device)
vit.eval()

# Load the trained effnet
eff = load_effnet(location2)

# Read in the style dictionary with predictions labels
style_dictionary = get_dictionary()

# Download user image
img = st.file_uploader(label = 'Upload Image',type = ['jpg','png'])

if img != None:
    

    # Display the image
    user_img = Image.open(img)
    st.image(user_img)
    st.title('Related Art Styles:')
    
    # ViT Predictions

    vit_tensor = vit_transform(user_img).unsqueeze(0)
    vit_tensor = vit_tensor.to(device)
    with torch.no_grad():
        vit_output = vit(vit_tensor)
        vit_output = softmax_torch(vit_output)
        vit_predictions,vit_index = torch.topk(vit_output,TOPK)
    vit_predictions = vit_predictions.cpu().detach().numpy()[0]
    vit_index = vit_index.cpu().detach().numpy()[0]
  
    # EffNet Predictions

    eff_tensor = eff_transform(user_img)

    

    eff_output = eff.predict(eff_tensor)

    eff_output = eff_output[0]
    eff_output = softmax(eff_output)
    eff_index = np.argsort(eff_output)
    eff_index = eff_index[-5:]
    eff_index = np.flip(eff_index)
    eff_predictions = eff_output[eff_index]
  
  # Displaying predictions
    cols = st.columns(2)
    with cols[0]:
        st.header('Predictions From ViT:')
        for idx,vit_pred in enumerate(vit_index):
            
            vit_style = style_dictionary[str(vit_pred)]
            vit_link_style = vit_style.replace(" ","+")
            vit_url = "https://www.google.com/search?q="+vit_link_style
            vit_text='[{style}]({link}) with {:.2f} percent certainty'.format(100*vit_predictions[idx],link=vit_url,style = vit_style )
            st.write(vit_text)
    with cols[1]:
        st.header('Predictions From Effnetv2')
        for idx,eff_pred in enumerate(eff_index):
        
            eff_style = style_dictionary[str(eff_pred)]
            eff_link_style = eff_style.replace(" ","+")
            eff_url = "https://www.google.com/search?q="+eff_link_style
            eff_text='[{style}]({link}) with {:.2f} percent certainty'.format(100*eff_predictions[idx],link=eff_url,style = eff_style )
            st.write(eff_text)
       
  
