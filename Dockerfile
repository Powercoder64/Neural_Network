FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y sudo

RUN adduser --disabled-password --gecos '' ubuntu
RUN adduser ubuntu sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ubuntu

SHELL ["/bin/bash", "-c"]

RUN sudo apt-get -qq install curl vim git zip

WORKDIR /home/ubuntu/

RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash ./Miniconda3-latest-Linux-x86_64.sh -b
ENV PATH="/home/ubuntu/miniconda3/bin:$PATH"
RUN echo ". /home/ubuntu/miniconda3/etc/profile.d/conda.sh" >> ~/.profile
RUN conda config --set ssl_verify no
RUN conda init bash
RUN conda config --set auto_activate_base false
RUN conda create -n BASNET python=3.8 --yes
SHELL ["conda", "run", "-n", "BASNET", "/bin/bash", "-c"]
RUN conda install -y pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
RUN pip install joblib==0.13.0
RUN pip install pandas==0.23.4
RUN pip install scikit-learn==0.20.0
RUN pip install scipy==1.1.0
RUN pip install tensorboard==1.15.0
RUN pip install tensorboard-logger==0.1.0
RUN pip install tensorflow==1.15.4
RUN pip install tensorflow-estimator==1.13.0
RUN pip install tqdm==4.31.1
RUN pip install opencv-python==4.5.5.62
RUN pip install python-csv==0.0.13
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "BASNET"]
