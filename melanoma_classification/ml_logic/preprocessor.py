'''
This module will normalize the images and split the data
into train, validation and test sets.
'''

from sklearn.model_selection import train_test_split
from melanoma_classification.params import *
import cv2
import tqdm

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
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_val, y_train, y_val

def preprocess_imgs(img_path) -> np.ndarray:
    '''
    Preprocess the input image
    '''
    X = []
    # Load all images in the folder img_path
    for img in tqdm(os.listdir(img_path)):
        try:
            img_array = cv2.imread(os.path.join(img_path, img))
            img_array = cv2.resize(img_array, (224, 224))  # Resize images
            X.append(img_array)
        except Exception as e:
            pass  # in case of a problem, skip this image
    img_array = np.array(X)
    return img_array / 255.

def preprocess_img(img_path) -> np.ndarray:
    '''
    Preprocess the input image
    '''
    try:
        img_array = cv2.imread(img_path)
        # print (f'Shape of image before resizing :{img_array.shape}')
        img_array = cv2.resize(img_array, (224, 224))  # Resize images
        # print (f'Shape of image after resizing :{img_array.shape}')

    except Exception as e:
        pass  # in case of a problem, skip this image

    return img_array / 255.
