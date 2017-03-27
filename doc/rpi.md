### Coin recognition on Raspberry PI
---
The models trained with the coin recognition data set can be easily deployed on a Raspberry PI. For this you only need one of the `*.hdf5` files containing the optimal model weights - you can find those under `/data/models/<model_name>`.

### How to run
The easiest way to run the demo on a RPI is in a container. You can use Docker or even Kubernetes for this.

#### Docker
You can use a pre-built container and run it on your raspberry with:

```
docker run --device /dev/video0 -v /dev:/dev:rw -v /sys/class/gpio:/sys/class/gpio:rw  -v /sys/bus:/sys/bus:rw --privileged -d taimir93/rpi-tensorflow python /notebooks/deep/coin_recognition/demo_rpi.py
```

#### Kubernetes
You can also deploy the small demo in a k8s cluster:

```
kubectl apply -f demo_rpi.yaml
```

### How to build
Alternatively you can build your own RPI container from scratch. Execute the following steps on your RPI device (or on a ARM server):

* copy your desired model's `*.hdf5` file to the `build` directory, naming it `best_model.hdf5`:

```
cp /path/to/trained_model.hdf5 <this_project>/build/best_model.hdf5
```


* copy your desired audio file (that should play when a coin is recognized by the camera) under `build`, name it `voice.mp3`:

```
cp /path/to/audio_file.mp3 <this_project>/build/voice.mp3
```

* build the docker container image:

```
cd <this_project>/build
docker build -t demo-rpi .
```

* you can now run the image directly with Docker and the above command or upload it to DockerHub and use it in the k8s `demo_rpi.yaml` file to deploy your model to a RPI device in a k8s cluster.
