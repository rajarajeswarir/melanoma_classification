'''
This module will normalize the images and split the data
into train, validation and test sets.
'''

from sklearn.model_selection import train_test_split
from melanoma_classification.params import *

def preprocess_data(X, y,test_size=0.2,
                      val_size=0.2) -> (np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray):

    '''
    Preprocess the inputs and split the data into train,
    validation and test sets.
    '''
    # Normalize images
    X = X / 255.


    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test
