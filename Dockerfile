FROM python:3.7-buster

ADD . /letr
WORKDIR /letr

RUN apt-get update && apt-get install -y --no-install-recommends tzdata g++ git curl nano

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt --upgrade --force-reinstall

# for CPU docker
RUN pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html