#!/usr/bin/env python
# coding: utf-8

import os
from os import path
import subprocess
import argparse
from importlib import import_module

parser = argparse.ArgumentParser()
parser.add_argument("model",
                    help="Path to the pretrained captioning model.")
parser.add_argument("images",
                    help="Folder with images to caption.")


# init paths
REPO_DIR = os.environ["ARTEMIS_DIR"]
PREPROCESSED_DATA = path.join(REPO_DIR, "preprocessed_artemis")
SAMPLE_SPEAKER = path.join(REPO_DIR, "artemis", "scripts", "sample_speaker.py")
IMG2EMO = path.join(REPO_DIR, "pretrained_models", "img2emo", "img2emo.pt") 

# # Initialize relevant directories

#main ARTEMIS directory
# artemis_repo_dir = os.environ["ARTEMIS_DIR"]

# where is the sample_speaker.py script?
# sample_speaker_script = os.path.join(artemis_repo_dir,"artemis", "scripts", "sample_speaker.py")

#path to ArtEmis/COCO preprocessed data
# preprocessed_data_dir = os.path.join(artemis_repo_dir, "preprocessed_artemis")

#where is the pretrained model?
model_path = os.path.join(artemis_repo_dir, "pretrained_models", "emo_grounded_model", "checkpoints","best_model.pt")

#where is the image to emotion classifier?
#img2emo = os.path.join(artemis_repo_dir, "pretrained_models", "img2emo","img2emo.pt")

#parent folder for this script
custom_caption_dir = os.path.join(artemis_repo_dir, "custom_scripts","emo_caption_custom_images")

# this file will be input for emotion detector
custom_emo_csv = os.path.join(custom_caption_dir,"temporary_files","custom_emo_images.csv")

# this file will be input for the captioner
custom_img_csv = os.path.join(custom_caption_dir,"temporary_files","custom_images.csv")

# where are the images to caption?
img_dir= os.path.join(custom_caption_dir, "images")

# where to pickle the results?
pickle_out_file = os.path.join(custom_caption_dir,"temporary_files","pickled_data.pkl")

#for logs
log_dir = os.path.join(custom_caption_dir, "temporary_files","logs")

#configuration file for sample_speaker.py
configuration_file_path = os.path.join(artemis_repo_dir, "pretrained_models","emo_grounded_model","config.json.txt")

#final captions will be here:
csv_out_file = os.path.join(custom_caption_dir, "outputs","custom_captions.csv")


# # Construct the CSV for custom images

# The `sample_speaker.py` script needs a csv of one column with header `image_file` and whose rows are the absolute paths of the images to be captioned.

#get image paths
filenames = [os.path.join(img_dir, filename) for filename in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, filename))]
filenames


# In[20]:


############ write the CSV for emotion detection
with open(custom_emo_csv, 'w') as custom_csv:
    custom_csv.write(",".join(['image_file'])) #header
    custom_csv.write('\n')
    #file paths:
    for filename in filenames:
        custom_csv.write(",".join([filename]))
        custom_csv.write('\n')


############ emotion detection

    #load the model and utilities
from artemis.in_out.neural_net_oriented import image_transformation
from artemis.in_out.neural_net_oriented import torch_load_model
from artemis.emotions import IDX_TO_EMOTION

import pandas as pd
import numpy as np
from PIL import Image
import torch

# device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu") # it is not working with gpus, only cpu :/
device =torch.device("cpu")
model = torch_load_model(img2emo, map_location=device)   
transformation = image_transformation(255)['train']

emotions_df = { # after emotion detection, to be conferted to a dataframe and saved as csv
    'image_file':[], #filenames
    'grounding_emotion':[], #main emotion
}

for i in range(len(IDX_TO_EMOTION)): # for ranked emotions
    emotions_df['emo' + str(i) ] = []
    
    #load the filenames
filename_df = pd.read_csv(custom_emo_csv)
filenames = filename_df['image_file']


    # go through each image and classify it
for filename in filenames:
    with Image.open(filename).convert('RGB') as img:

        img = transformation(img).unsqueeze(0)# unsqueeze to add artificial first dimension
        
        # emotion detection
        emotion_vector = model(img) # apply the model
        emotion_vector = np.exp(emotion_vector.detach().numpy()) # calculate probabilities
        sorted_indices = (-emotion_vector).argsort()[0] # sort from most to least likely
        emotions = [IDX_TO_EMOTION[ind] for ind in sorted_indices]

        #construct the csv line
        emotions_df['image_file'].append(filename)
        emotions_df['grounding_emotion'].append(emotions[0])
        for i in range(len(emotions)):
            emotions_df['emo' + str(i)].append(emotions[i])



############ save detected emotions to a CSV for captioning
pd.DataFrame(emotions_df).to_csv(custom_img_csv)
"""
with open(custom_img_csv, 'w') as custom_csv:
#    custom_csv.write(",".join(['image_file'])) #header
    custom_csv.write(",".join(['image_file','grounding_emotion', 'emotion','test_1'])) #header
    custom_csv.write('\n')
    #file paths:
    for filename in filenames:
#        custom_csv.write(",".join([filename]))
        custom_csv.write(",".join([filename, "awe", "anger", "there"]))
        custom_csv.write('\n')
"""



# # Edit configuration file
# 
# This file is in ARTEMIS_DIR/pretrained_models/emo_grounded_model/config.json.txt and it is a JSON file with parameters for the captioning algorithm. 
# I think the program should be modified to accept them in the command line.

# In[21]:


import json

with open(configuration_file_path) as configuration_file:
    configuration_json = json.load(configuration_file)

configuration_json["data_dir"] = preprocessed_data_dir
configuration_json["log_dir"] = log_dir

with open(configuration_file_path, "w") as configuration_file:
    json.dump(configuration_json, configuration_file)


# # Caption

# In[26]:


process = subprocess.run(
    [
    "python3",
    sample_speaker_script,
    "-speaker-saved-args",
    configuration_file_path,
    "-speaker-checkpoint",
    model_path,
    #"-data-dir",
    #preprocessed_data_dir,
    "-img-dir" ,
    img_dir,
    "-out-file" ,
    pickle_out_file,
    #'-log-dir',
    #log_dir,
    "--custom-data-csv",
    custom_img_csv
    ],
    stdout=subprocess.PIPE
)


# # Process pickled output into CSV

results_df = pd.read_pickle(pickle_out_file)

#add the detected emotions
for i in range(len(IDX_TO_EMOTION)):
    results_df['emo'+ str(i)] = emotions_df['emo'+ str(i)]

results_df.to_csv(csv_out_file)
