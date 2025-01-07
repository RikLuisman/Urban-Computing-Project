## A Foundational Approach for Urban-Scale Segmentation
### Leiden University ~ Urban Computing ~ Final Project 
*by Anca Matei (s4004507) and Rik Luisman (s2484956)*

This project contains a pipeline for finetuning Prithvi, using two different datasets: one with satellite imagery (DOTA) and one with aerial imagery (LoveDA). 
The repository also contains the scripts necessary to finetune the UNetFormer model, used as a benchmark for our study. 

### Requirements
1. pip install -r requirements.txt


### Overview 

The pipeline for fine-tuning Prithvi (and UNetFormer) consists of three main components:

(1) Preprocessing Scripts

(2) Fine-tuning Scripts

(3) Evaluation Scripts

*The repository includes auxiliary scripts for two types of visualizations: one for original images with their annotations and another for the results.

### Directories explanations
*Disclaimer: The directories used in this project are not included in the repository due to their large size. However, their contents can be replicated, with the exception of those containing the original images.*

1. Two **datasets directories** (the links to download the datasets are provided in our report; however, the datasets can also be made available to the user upon request.)
2. Two directories for the **preprocessed datasets** (one for Prithvi and one for UNetFormer). Each directory contains the preprocessed images and masks, organized into train, validation, and test sets. 

Paths examples:\
Preprocessed Datasets Prithvi/DOTA/train/images/P0000.tif\
Preprocessed Datasets Prithvi/DOTA/train/masks/P0000.tif

3. Three directories with the **fine-tuned models** (two for Prithvi and one for UNetFormer)

Paths examples:\
FineTuned_Prithvi_models_DOTA/best_segmentation_model_5epochs_0.001.pth\
FineTuned_Prithvi_models_LoveDa/rural/best_segmentation_model_5epochs_0.001.pth
FineTuned_UNetFormer_models_DOTA/best_segmentation_model_5epochs_0.001.pth

 
### Running 
In order to replicate the files in the directories you need to run:\
`preprocess_{ModelName}_{DatasetName}_dataset.py `\
`finetuning_{ModelName}\_for_{DatasetName}.py` -> set the number of epochs and the learning rate

In order to test the fine-tuned models you need to run: \
`evaluation_{ModelName}_{DatasetName}.py` -> set the number of epochs and the learning rate to obtain the correct path for the saved model

To view an original image with the categories overlaid on top, run the following command:\
`visualization_{DatasetName}_dataset.py`

To generate the bar plots featured in our report, comparing the results of Prithvi and UNetFormer, run:\
`comparative_visualization_results_{DatasetName}.py`

### Additional information

For each model there are a few original scripts used for the fine-tuning process, which were downloaded from online sources/repositories.

*Disclaimer: The `Prithvi_100M.pt` file, containing the pretrained weights of the model, is quite large and has also been added to the .gitignore. If needed, it can be provided upon request by contacting us.*

----------------------------

#### Contact information
a.matei.2@umail.leidenuniv.nl\
r.t.luisman@umail.leidenuniv.nl