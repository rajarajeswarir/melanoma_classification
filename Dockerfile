
# '''
# #FROM tensorflow/tensorflow:2.10.0
# FROM python:3.8.12-buster

# #WORKDIR /prod


# COPY requirements.txt requirements.txt
# COPY melanoma_classification melanoma_classification


# RUN pip install -r requirements.txt
# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# #RUN make reset_local_files

# # You can add --port $PORT if you need to set PORT as a specific env variable
# CMD uvicorn melanoma_classification.api.fast:app --host 0.0.0.0 --port $PORT
# '''


FROM python:3.8.12-buster
# WORKDIR /api
# ENV LOCAL_REGISTRY_PATH="api/training_outputs"

RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY melanoma_classification melanoma_classification
COPY training_outputs training_outputs

RUN pip install --no-cache-dir -r requirements.txt

# EXPOSE 8000
# You can add --port $PORT if you need to set PORT as a specific env variable
CMD uvicorn melanoma_classification.api.fast:app --host 0.0.0.0 --port $PORT
