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
RUN git clone --depth 1 https://github.com/raspberrypi/userland.git /deep/userland
RUN cd /deep/userland && /deep/userland/buildme
RUN ldconfig
ENV LD_LIBRARY_PATH="/opt/vc/lib"

# install pygame (for camera interactions)
RUN apt-get install -y python-pygame

# install other python dependencies
RUN pip install pyyaml

# Manually install keras
RUN mkdir -p /root/.keras
COPY keras.json /root/.keras/keras.json
RUN git clone https://github.com/fchollet/keras.git /deep/keras
RUN cd /deep/keras && git checkout 1.2.2
RUN cp -R /deep/keras/keras /usr/local/lib/python2.7/dist-packages


COPY best_model.hdf5 /deep/best_model.hdf5
COPY voice.mp3 /deep/voice.mp3
COPY demo_rpi.py /deep/demo_rpi.py
COPY data_mean.npy /deep/data_mean.npy

