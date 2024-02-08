#FROM tensorflow/tensorflow:2.10.0
FROM python:3.8.12-buster

#WORKDIR /prod


COPY requirements.txt requirements.txt
COPY melanoma_classification melanoma_classification


RUN pip install -r requirements.txt


#RUN make reset_local_files

# You can add --port $PORT if you need to set PORT as a specific env variable
CMD uvicorn melanoma_classification.api.fast:app --host 0.0.0.0 --port $PORT
