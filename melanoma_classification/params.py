
import os
import numpy as np

##################  VARIABLES  ##################
DATA_SIZE = "all" # ["500", "1k", "2k", "all"]
CHUNK_SIZE = 200 # Not used for the moment
GCP_PROJECT = "melanoma-classification" # TO COMPLETE
BQ_REGION = "EU" # TO COMPLETE
MODEL_TARGET = "local" # ["local", "gcp"]
DATA_SOURCE = "local" # ["local", "gcp"]
GCP_BUCKET_NAME = "sumitkamra20-melanoma-images" # TO COMPLETE

# IMAGE=melanoma_classification
##################  CONSTANTS  #####################

LOCAL_DATA_PATH = '/Users/sumitkamra/code/rajarajeswarir/melanoma_classification/melanoma_cancer_dataset'
# LOCAL_REGISTRY_PATH =  "/Users/sumitkamra/code/rajarajeswarir/melanoma_classification/training_outputs"
LOCAL_REGISTRY_PATH =  "/training_outputs"
