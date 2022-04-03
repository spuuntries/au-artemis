#!/usr/bin/env python3
# coding: utf-8

import os
import subprocess
import argparse
from importlib import import_module


# argument parser
parser = argparse.ArgumentParser("captioner")
parser.add_argument("config",
					help="The emo-grounded-model configuration file.")
parser.add_argument("imgdir",
					help="Images to be captioned directory.")
parser.parse_args()


# init paths
REPO_DIR = os.environ["ARTEMIS_DIR"]


def create_csv(img_dir, ):
	"""Create the CSV for emotion detection."""
	
	img_names = list(filter(os.path.isfile,
				[os.path.join(img_dir, fname) for fname in os.listdir(img_dir)]))

	with open(filename, 'w+') as custom_csv:
		custom_csv.write("image_file\n") # header
		for fname in img_names:
			custom_csv.write(f"{fname}\n")


def detect_emotion(img_name):
	# preprocessing
	with Image.open(img_name).convert('RGB') as img:
		img = transformation(img).unsqueeze(0) # unsqueeze to add articficial 1st dimension

		# emotion detection
		emotion_vector = model(img) # apply the model
		

def detect_emotions()
	"""Load img2emo model and retrieve emotions."""
	# init
	device = torch.device("cpu")
	model = torch_load_model(img2emo, map_location=device)
	transformation = image_transformation(255)['train']


if __name__=="__main__":
	detect_emotions()
	process = subprocess.run(
	    [
	    "python3", sample_speaker_script,
	    "-speaker-saved-args", configuration_file_path,
	    "-speaker-checkpoint", model_path,
	    #"-data-dir", #preprocessed_data_dir,
	    "-img-dir" , img_dir,
	    "-out-file" , pickle_out_file,
	    #'-log-dir', #log_dir,
	    "--custom-data-csv", custom_img_csv
	    ],
	    stdout=subprocess.PIPE
	)