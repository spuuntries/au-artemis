#!/usr/bin/env python
# coding: utf-8

# In[14]:


import os
import subprocess


# # Initialize elevant directories

# In[ ]:


#main ARTEMIS directory
artemis_repo_dir = os.environ["ARTEMIS_DIR"]

# where is the sample_speaker.py script?
sample_speaker_script = os.path.join(artemis_repo_dir,"artemis", "scripts", "sample_speaker.py")

#path to ArtEmis/COCO preprocessed data
preprocessed_data_dir = os.path.join(artemis_repo_dir, "preprocessed_artemis")

#where is the pretrained model?
model_path = os.path.join(artemis_repo_dir, "pretrained_models", "emo_grounded_model", "checkpoints","best_model.pt")


#parent folder for this script
custom_caption_dir = os.path.join(artemis_repo_dir, "custom_scripts","emo_caption_custom_images")


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

# In[ ]:


#get image paths
filenames = [os.path.join(img_dir, filename) for filename in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, filename))]
filenames


# In[20]:


#write the CSV
with open(custom_img_csv, 'w') as custom_csv:
    custom_csv.write('image_file') #header
    custom_csv.write('\n')
    #file paths:
    for filename in filenames:
        custom_csv.write(filename)
        custom_csv.write('\n')


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

# In[ ]:


import pickle
#from IPython.display import display
import pandas as pd

#from IPython.display import display
#from PIL import Image

df = pd.read_pickle(pickle_out_file)
df.to_csv(csv_out_file)
