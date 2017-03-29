FROM romilly/rpi-docker-tensorflow

RUN apt-get update && apt-get install -y \
	python-dev \
	python-rpi.gpio \
	python-h5py \
	git \
	libjpeg-dev \
	libfreetype6 \
	libfreetype6-dev \
	zlib1g-dev \
	omxplayer \
	cmake \
	build-essential

# Install userland
RUN git clone --depth 1 https://github.com/raspberrypi/userland.git /home/deep/userland
RUN cd /home/deep/userland && /home/deep/userland/buildme
ENV LD_LIBRARY_PATH="/opt/vc/lib"
RUN ldconfig

# install pygame (for camera interactions)
RUN apt-get install -y python-pygame

# install other python dependencies
RUN pip install pyyaml

# Manually install keras
RUN mkdir -p /home/deep/.keras
COPY keras.json /home/deep/.keras/keras.json
RUN git clone https://github.com/fchollet/keras.git /home/deep/keras
RUN cd /home/deep/keras && git checkout 1.2.2
RUN cp -R /home/deep/keras/keras /usr/local/lib/python2.7/dist-packages


COPY best_model.hdf5 /home/deep/best_model.hdf5
COPY voice.mp3 /home/deep/voice.mp3
COPY demo_rpi.py /home/deep/demo_rpi.py

# RUN useradd -ms /bin/bash deep
# RUN adduser deep sudo
# RUN echo deep:deep | chpasswd
# USER deep
