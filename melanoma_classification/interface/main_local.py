'''
This file will be used to test the model locally.
'''

# Import modules in melanoma_classification

from colorama import Fore, Style
from melanoma_classification.ml_logic.model import initialize_model, compile_model
from melanoma_classification.ml_logic.registry import save_model, save_results, load_model
from melanoma_classification.ml_logic.preprocessor import preprocess_data
from melanoma_classification.ml_logic.data import load_data, load_and_save_images
from melanoma_classification.ml_logic.model import evaluate_model, train_model, predict_results


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

    print(Fore.BLUE + "\nPredicting model..." + Style.RESET_ALL)
    image_path = '/Users/sumitkamra/code/rajarajeswarir/melanoma_classification/image_for_prediction/test_file.jpeg'
    results = predict_results(image_path)

    print(results)
