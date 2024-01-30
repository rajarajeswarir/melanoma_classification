'''
This file contains the functions to load image data from local or GCP
and save the data in npy format to disk for later use.
'''


from pathlib import Path
import os
import cv2
import numpy as np
from tqdm import tqdm  # for progress bar

from melanoma_classification.params import *
from tensorflow.keras.utils import to_categorical

def load_images_and_labels(directory) -> (np.ndarray, np.ndarray):
    '''
    Load images and labels from a directory
    and return them as numpy unshuffledz arrays.
    '''

    X = []
    y = []

    for label in ['malignant', 'benign']:
        path = os.path.join(directory, label)
        class_num = 1 if label == 'malignant' else 0

        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.resize(img_array, (224, 224))  # Resize images
                X.append(img_array)
                y.append(class_num)
            except Exception as e:
                pass  # in case of a problem, skip this image
    y = np.array(y)
    y = to_categorical(y)
    return np.array(X), y


def load_and_save_images(data_source=DATA_SOURCE):
    '''
    Load the image data from local or GCP and save the data in npy format to disk for later use.
    '''
    if data_source == "local":
        data_path = LOCAL_DATA_PATH
        train_dir = os.path.join(data_path, 'train')
        test_dir = os.path.join(data_path, 'test')

        X_train, y_train = load_images_and_labels(train_dir)
        X_test, y_test = load_images_and_labels(test_dir)


    elif data_source == "gcp":
        data_path = GCP_BUCKET_NAME
        print("GCP not implemented yet")
    else:
        raise ValueError("data_source must be either 'local' or 'gcp'")

    # Shuffle the data
    np.random.seed(42)
    train_shuffle = np.random.permutation(len(X_train))
    test_shuffle = np.random.permutation(len(X_test))
    X_train = X_train[train_shuffle]
    y_train = y_train[train_shuffle]
    X_test = X_test[test_shuffle]
    y_test = y_test[test_shuffle]

    # Save the data
    train_test_data_path = os.path.join(LOCAL_DATA_PATH, 'train_test_data')
    os.makedirs(train_test_data_path, exist_ok=True)
    np.save(os.path.join(train_test_data_path, "X_train.npy"), X_train)
    np.save(os.path.join(train_test_data_path, "y_train.npy"), y_train)
    np.save(os.path.join(train_test_data_path, "X_test.npy"), X_test)
    np.save(os.path.join(train_test_data_path, "y_test.npy"), y_test)

def load_data(data_size = DATA_SIZE) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    '''
    Load the data from the npy files and return them.
    '''
    train_test_data_path = os.path.join(LOCAL_DATA_PATH, 'train_test_data')
    X_train = np.load(os.path.join(train_test_data_path, "X_train.npy"))
    y_train = np.load(os.path.join(train_test_data_path, "y_train.npy"))
    X_test = np.load(os.path.join(train_test_data_path, "X_test.npy"))
    y_test = np.load(os.path.join(train_test_data_path, "y_test.npy"))

    if data_size == "500":
        X_train = X_train[:500]
        y_train = y_train[:500]
        X_test = X_test[:500]
        y_test = y_test[:500]
    elif data_size == "1k":
        X_train = X_train[:1000]
        y_train = y_train[:1000]
        X_test = X_test[:1000]
        y_test = y_test[:1000]
    elif data_size == "all":
        pass
    else:
        raise ValueError("data_size must be either '500', '1k' or 'all'")

    return X_train, y_train, X_test, y_test
