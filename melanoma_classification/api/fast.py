'''
The file which contains api logic for fastapi.
'''

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
import cv2
import os
import numpy as np
from melanoma_classification.ml_logic.model import load_this_model


from melanoma_classification.ml_logic.model import load_model, predict_results

##for single image
app = FastAPI()

# Allowing all middleware is optional but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),  # File upload for image
):
    try:
        file_name = image.filename
        file_size = image.file.__sizeof__()
        file_type = image.content_type
        data_type = type(image)

        # Read the image file
        # img_data = cv2.imread(image.file.read())
        # image_data = Image.open(BytesIO(image.file.read()))

        image_data = await image.read()
        im_data = Image.open(BytesIO(image_data))
        # im_data = cv2.imread(image.file.read())

        # test_file_path = '/Users/sumitkamra/code/rajarajeswarir/melanoma_classification/image_for_prediction'
        # im_data.save(os.path.join(test_file_path, 'test_file.jpeg'))

        img_array = np.array(im_data)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        img_array = cv2.resize(img_array, (224, 224)) / 255.
        img_array = np.expand_dims(img_array, axis=0)
        model = load_this_model("all_20240212-115256.h5")
        results = model.predict(img_array, verbose=1)

        # results = predict_results('/Users/sumitkamra/code/rajarajeswarir/melanoma_classification/image_for_prediction/test_file.jpeg')

        outcome = {'malignant_probability': float(results[0][1]),
                'benign_probability': float(results[0][0])}

        '''
        # Convert the image to the required format (resize, normalize,)
        # processed_image = preprocess_data(image_data)
        # Reshape the image to match the input shape of the model
        #processed_image = processed_image.reshape((1,) + processed_image.shape)

        # Make predictions using the loaded model
        results = predict_results(img_data)

        # Extract the result from the prediction
        outcome = {'malignant_probability': float(results[0][1]),
                'benign_probability': float(results[0][0])}'''
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=422, detail="Image file is required")

    # Return the prediction in the specified JSON format
    # return {'file_name': file_name,
    #         'file_size': file_size,
    #         'file_type': file_type,
    #         'data_type': str(data_type),
    #         'im_type': str(im_type)
    #         }
    return {'outcome': outcome}

@app.get("/")
async def root():
    return {'greeting': 'Hello'}
