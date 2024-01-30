'''
This module will contain the model architecture
and training logic.
'''
from colorama import Fore, Style
import time
import matplotlib.pyplot as plt

# Timing the TF import

print(Fore.BLUE + "\nLoading TensorFlow and VGG16..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow.keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")

def initialize_model(train_weights=False, input_shape: tuple = (224, 224, 3)) -> Model:
    '''
    Initialize the model.
    '''
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = train_weights

    # Add flatten and dense layers on top of VGG16
    flatten_layer = layers.Flatten()
    dense_layer_1 = layers.Dense(50, activation='relu')
    dense_layer_2 = layers.Dense(20, activation='relu')
    prediction_layer = layers.Dense(2, activation='softmax')

    vgg_model = Sequential([
        base_model,
        flatten_layer,
        dense_layer_1,
        dense_layer_2,
        prediction_layer
    ])
    print("✅ Model initialized")

    return vgg_model

def compile_model(model: Model,
                  learning_rate=0.0005,
                  metrics = 'accuracy') -> Model:
    """
    Compile the Neural Network
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=[metrics])

    print("✅ Model compiled")

    return model

def train_model(model: Model,
                X_train, y_train,
                X_val, y_val,
                epochs=10,
                batch_size=32,
                callbacks=[],
                verbose=1) -> Model:
    """
    Train the Neural Network
    """
    es = EarlyStopping(monitor='val_accuracy',
                       mode='max', patience=2,
                       restore_best_weights=True)


    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[es],
                        verbose=verbose)

    print(f"✅ Model trained on {len(X_train)} images")
    print(f"✅ Model validation accuracy: {round(history.history['val_accuracy'][-1], 2)}")

    return model, history

def evaluate_model(model: Model,
                     X_test, y_test,
                     verbose=1) -> tuple:
    """
    Evaluate the model on the test set
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=verbose)

    print(f"✅ Model test accuracy: {round(accuracy, 2)}")

    return loss, accuracy

def plot_history(history, title='', axs=None, exp_name=""):
    if axs is not None:
        ax1, ax2 = axs
    else:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if len(exp_name) > 0 and exp_name[0] != '_':
        exp_name = '_' + exp_name
    ax1.plot(history.history['loss'], label='train' + exp_name)
    ax1.plot(history.history['val_loss'], label='val' + exp_name)
    #ax1.set_ylim(0., 2.2)
    ax1.set_title('loss')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='train accuracy'  + exp_name)
    ax2.plot(history.history['val_accuracy'], label='val accuracy'  + exp_name)
    #ax2.set_ylim(0.25, 1.)
    ax2.set_title('Accuracy')
    ax2.legend()
    return (ax1, ax2)
