FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get install -y libgl1

WORKDIR /code

ENV PYTHONUNBUFFERED 1

COPY requirements.txt .
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install -r requirements.txt

COPY . .
ENTRYPOINT python train.py
