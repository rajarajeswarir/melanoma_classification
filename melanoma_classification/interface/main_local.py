'''
This file will be used to test the model locally.
'''

# Import modules in melanoma_classification

from colorama import Fore, Style
from melanoma_classification.ml_logic.model import initialize_model, compile_model
from melanoma_classification.ml_logic.preprocessor import preprocess_data
from melanoma_classification.ml_logic.data import load_data, load_and_save_images
from melanoma_classification.ml_logic.model import evaluate_model, train_model



if __name__ == "__main__":

    print(Fore.BLUE + "\nReading images and savig them locally..." + Style.RESET_ALL)
    load_and_save_images()

    print(Fore.BLUE + "\nLoading data..." + Style.RESET_ALL)
    X_train, y_train, X_test, y_test = load_data()

    print(Fore.BLUE + "\nPreprocessing data..." + Style.RESET_ALL)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(X_train, y_train)

    print("------------------------------------------------------------------")
    print(Fore.BLUE + "\nX_train shape:" + Style.RESET_ALL, X_train.shape)
    print(Fore.BLUE + "\nX_val shape:" + Style.RESET_ALL, X_val.shape)
    print(Fore.BLUE + "\nX_test shape:" + Style.RESET_ALL, X_test.shape)
    print(Fore.BLUE + "\ny_train shape:" + Style.RESET_ALL, y_train.shape)
    print(Fore.BLUE + "\ny_val shape:" + Style.RESET_ALL, y_val.shape)
    print(Fore.BLUE + "\ny_test shape:" + Style.RESET_ALL, y_test.shape)
    print("------------------------------------------------------------------")



    print(Fore.BLUE + "\nInitializing model..." + Style.RESET_ALL)
    model = initialize_model()

    print(Fore.BLUE + "\nCompiling model..." + Style.RESET_ALL)
    model = compile_model(model)

    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)
    model, history = train_model(model, X_train, y_train, X_val, y_val)

    print(Fore.BLUE + "\nEvaluating model..." + Style.RESET_ALL)
    evaluate_model(model, X_test, y_test)
