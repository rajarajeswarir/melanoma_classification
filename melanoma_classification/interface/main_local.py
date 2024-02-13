'''
This file will be used to test the model locally.
'''

# Import modules in melanoma_classification

from colorama import Fore, Style
from melanoma_classification.ml_logic.model import initialize_model, compile_model
from melanoma_classification.ml_logic.registry import save_model, save_results, load_model, load_this_model
from melanoma_classification.ml_logic.preprocessor import preprocess_data
from melanoma_classification.ml_logic.data import load_data, load_and_save_images
from melanoma_classification.ml_logic.data import load_test_data, load_and_save_resnet_test
from melanoma_classification.ml_logic.model import evaluate_model, train_model, predict_results
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator

def evaluate_this_model(mod, name = "Name not provided"):
    X_test, y_test = load_test_data()
    model = load_this_model(mod)

    print(Fore.CYAN + f"\nEvaluating model ({name})..." + Style.RESET_ALL)
    evaluate_model(model, X_test, y_test)

def evaluate_resnet_model(mod):
    X_test, y_test = load_test_data(resnet=True)
    resnet_model = load_this_model(mod)
    print(Fore.BLUE + f"\nX_test shape:" + Style.RESET_ALL, X_test.shape)
    print(Fore.RED + f"\nEvaluating Resnet model)..." + Style.RESET_ALL)
    evaluate_model(resnet_model, X_test, y_test)

if __name__ == "__main__":
    '''
    print(Fore.BLUE + "\nReading images and savig them locally..." + Style.RESET_ALL)
    load_and_save_images()

    print(Fore.BLUE + "\nLoading data..." + Style.RESET_ALL)
    X_train, y_train, X_test, y_test = load_data()

    print(Fore.BLUE + "\nPreprocessing data..." + Style.RESET_ALL)
    X_train, X_val, y_train, y_val = preprocess_data(X_train, y_train)

    print("------------------------------------------------------------------")
    print(Fore.BLUE + "\nX_train shape:" + Style.RESET_ALL, X_train.shape)
    print(Fore.BLUE + "\nX_val shape:" + Style.RESET_ALL, X_val.shape)
    print(Fore.BLUE + "\nX_test shape:" + Style.RESET_ALL, X_test.shape)
    print(Fore.BLUE + "\ny_train shape:" + Style.RESET_ALL, y_train.shape)
    print(Fore.BLUE + "\ny_val shape:" + Style.RESET_ALL, y_val.shape)
    print(Fore.BLUE + "\ny_test shape:" + Style.RESET_ALL, y_test.shape)
    print("------------------------------------------------------------------")

    params = {
        'batch_size': 32,
        'patience': 2,
        'epochs': 10,
    }

    print(Fore.BLUE + "\nInitializing model..." + Style.RESET_ALL)
    model = initialize_model()

    print(Fore.BLUE + "\nCompiling model..." + Style.RESET_ALL)
    model = compile_model(model)

    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)
    model, history = train_model(model, X_train, y_train, X_val, y_val,
                                 epochs=params['epochs'],
                                 batch_size=params['batch_size'],
                                 patience=params['patience'])

    print(Fore.BLUE + "\nSaving results and models..." + Style.RESET_ALL)
    save_results(params=params, metrics=dict(mae=history.history['val_accuracy'][-1]))
    save_model(model=model)

    print(Fore.BLUE + "\nEvaluating model..." + Style.RESET_ALL)
    evaluate_model(model, X_test, y_test)
    '''

    '''
    print(Fore.BLUE + "\nPredicting model..." + Style.RESET_ALL)
    image_path = '/Users/sumitkamra/code/rajarajeswarir/melanoma_classification/image_for_prediction/melanoma_10113.jpg'
    results = predict_results(image_path)

    print(results)
    '''

    # Save images for resnet 256 / 256 format
    # load_and_save_resnet_test()

    '''
    mod_all_sumit = "all_20240212-115256.h5"
    mod_all_raji = "raji_20240212-130249.h5"
    mod_500_sumit = "20240203-080756.h5"
    mod_2k_sumit = "20240212-102341.h5"

    # evaluate_this_model(mod_500_sumit, "Data size 500")
    # evaluate_this_model(mod_2k_sumit, "Data size 2k")
    evaluate_this_model(mod_all_sumit, "Data size all")

    evaluate_resnet_model(mod_all_raji)
    '''
    # model = load_this_model("all_20240212-115256.h5")
    image_path = '/Users/sumitkamra/code/rajarajeswarir/melanoma_classification/image_for_prediction/melanoma_10113.jpg'
    # image_path = '/Users/sumitkamra/code/rajarajeswarir/melanoma_classification/image_for_prediction/melanoma_10113.jpg'
    results = predict_results(image_path)
