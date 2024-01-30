
import os
import numpy as np

##################  VARIABLES  ##################
DATA_SIZE = "500" # ["500", "1k", "all"]
CHUNK_SIZE = 200 # Not used for the moment
GCP_PROJECT = "melanoma-classification" # TO COMPLETE
BQ_REGION = "EU" # TO COMPLETE
MODEL_TARGET = "local" # ["local", "gcp"]
DATA_SOURCE = "local" # ["local", "gcp"]
GCP_BUCKET_NAME = "sumitkamra20-melanoma-images" # TO COMPLETE
##################  CONSTANTS  #####################

LOCAL_DATA_PATH = '/Users/sumitkamra/code/rajarajeswarir/melanoma_classification/melanoma_cancer_dataset'

'''
The constants below need to be updated with the values
'''
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs")
