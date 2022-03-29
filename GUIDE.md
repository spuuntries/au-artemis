## ArtEmis for the supervised project

This is another iteration of the installation guide made by Luis, redacted by Karolin.

### Setup

0. Open a terminal in this (=the top most artemis) directory.
1. Create and activate a virtual environment:
	```
	python3 -m venv artemis-venv
	source artemis-venv/bin/activate
	```
2. Install the artemis package:
	```
	pip install -e .
	```
3. After filling this [form](https://forms.gle/7eqiRgb764uTuexd7), you will get a zip file. Unpack it and you will get an `official_data` directory that contains three files:
- `artemis_dataset_release_v0.csv`
- `ola_dataset_release_v0.csv`
- `README.txt`
Move it to this directory.

4. Preprocess the provided annotations (it will take a while).

   ```
   python3 artemis/scripts/preprocess_artemis_data.py -save-out-dir preprocessed_artemis/ -raw-artemis-data-csv official_data/artemis_dataset_release_v0.csv --preprocess-for-deep-nets True
   ```

5. Download [this pretrained model](https://www.dropbox.com/s/0erh464wag8ods1/emo_grounded_sat_speaker_cvpr21.zip?dl=0), which is a folder called `03-17-2021-20-32-19`, rename it to `emo_ground_model` and place it in `pretrained_models/` directory. (So that you have a folder `pretrained_models/emo_grounded_model/`.)

   This folder should have the following structure:

   - README.txt
   - checkpoints/
   - config.json.txt
   - log.txt
   - samples_from_best_model_test_split.pkl
   - tb_log/

6. Also, download this [pretrained image-to-emotion classifier model](https://www.dropbox.com/s/8dfj3b36q15iieo/best_model.pt?dl=0), which is a file `best_model.pt`, rename it to `img2emo.pt` and place it in `pretrained_models/img2emo/`.

7. Congrats! :tada: Now you are ready to use the models.

### Usage

#### Emotion grounded speaker

This is a pretrained model that uses k-nearest neighbors algorithm to retrieve a caption and tries to detect emotion from the caption.

1. Fix the path: `export ARTEMIS_DIR=$PWD`

2. Don’t forget to activate the virtual environment: `source $ARTEMIS_DIR/artemis-venv/bin/activate`

3. Put the images you wish to caption in the directory `$ARTEMIS_DIR/custom_scripts/emo_caption_custom_images/images`. For good measure, avoid spaces in the filenames.

4. To caption, simply run:

   ```
   python3 $ARTEMIS_DIR/custom_scripts/emo_caption_custom_images/custom_captions.py
   ```

5. The result will be a CSV file in `$ARTEMIS_DIR/custom_scripts/emo_caption_custom_images/outputs` containing `(image_path, emotion, caption)`.

6. To look at some images with their captions, open the notebook `$ARTEMIS_DIR/custom_scripts/emo_caption_custom_images/custom_captions_visualize.ipynb`

7. Congrats again! :tada:



### Details

For some reason, the image-to-emotion classifier only runs on CPUs and not on GPUs. If I try to run it as GPY, I get a `tensor dimensions are not matching` error.

In `custom_captions.py` see the definition of the variable `device`:

```
device = torch.device("cpu")
```

