FROM pytorch/pytorch

VOLUME /data
VOLUME /results

RUN sudo apt-get install git
RUN pip install pillow tqdm h5py
RUN git clone https://github.com/qscgy/ultrasound
WORKDIR /ultrasound/

RUN git submodule update --init --recursive

CMD "/bin/bash/"