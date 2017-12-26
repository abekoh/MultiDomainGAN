FROM nvidia/cuda:8.0-cudnn6-runtime
RUN apt-get update
RUN apt-get -y install python3-dev python3-pip curl
RUN pip3 install -U pip
RUN pip3 install tensorflow-gpu tqdm h5py
