# getting docker base image
FROM dokken/ubuntu-20.04

RUN apt-get install -y software-properties-common

RUN add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get update

RUN apt-get install -y python3.10

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2

RUN update-alternatives --config python3

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

RUN python3 get-pip.py 

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/tirthankarCU/NumberLearningInChildren_ML.git

# changing mount point
WORKDIR /NumberLearningInChildren_ML

# installing python packages.
RUN pip install -r Env_creation/requirements0.txt

# installing additional python packages.
RUN pip install -r Env_creation/requirements1.txt

# for open cv2
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


