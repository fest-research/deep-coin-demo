# Coin recognition
---

This is a small demonstration of the capabilities of discriminative models in deep learning. In particular, small 200x200 pixels images are classified as "containing a 50 cent coin" or "not containing a 50 cent coin". The project is meant as a tutorial, illustrating very basic techniques in deep learning (and in particular ConvNets).

#### Dependencies
Check `requirements.txt` for pure python requirements. You can install them with:

```
pip install -r requirements.txt
```

You need `OpenCV` and its python wrapper, with video features enabled. For a pre-compiled version, run:

```
sudo apt-get install libopencv-dev python-opencv
```

Also, if `tkinter` is not installed on your system, you should do so with:

```
sudo apt-get install python-tk
```


#### Model configuration
You can configure basic hyper-parameters of the classifier in `config.yaml`. The configuration file is loaded from the file system every time `main.py` is started.

```yaml
---
data:
  n_channels: 3
  img_width: 200
  img_height: 200
model:
  name: wider_network
  n_classes: 2
training:
  dropout: 0.0
  learning_rate: 0.001
  batch_size: 32
  epochs: 200
  network: wider_network
```

In the example above, `model.name` is important - this is the ID of the experiment you are running. During training, a folder for this model will be created: `data/folder/<model_name>`, where all training results and artifacts (such as the `weights*.hdf5` files) will be stored.

#### Training
To start training the model, run:

```
python main.py
```

The training will start and all results will be stored in the model's directory, determined based on the `config.yaml` as described above. As the training progresses, the model weights will be checkpointed every time a new best is reached.

#### Testing
A small online demo was created to show the pre-trained classifier in action. A camera needs to be connected to your machine. Run:

```
python demo.py --modeldir=./data/models/<model_name> --weights=<weights_filename>.hdf5
```

You can find the name of the weights file you want by inspecting the contents of `data/models/<model_name>`. An example is `weights.255-0.02-0.99.hdf5`. Here 255 is the training epoch (iteration) when the weights were persisted, 0.02 is the validation loss value at that point (the lower, the better) and 0.99 is the validation accuracy (the bigger, the better).
A live feed from the camera will be processed frame by frame by the selected classifier and the frames will be classified as either "containing a coin" or "not containing a coin" on the fly.

#### Data management
In order to train the model, attention should be paid to the structure of the `data` folder:

* `data/positive` should contain 200x200 images with a coin in them. For this, you can extract `positive.tar.gz`.
* `data/negative` should contain 200x200 images that have no coin in them. For this, you can extract `negative.tar.gz`.
* `data/models` will contain the results of all trained models; you do not need to create this directory manually

Additionally, you can generate new data samples from a camera video feed using this script:

```
python -m fujitsu.data_management.data_generator
```

Note that all images will be stored in `data`, and then you need to manually sort them into `data/positive` or `data/negative`, depending on whether they contain a coin (positive) or not.

### Visualizing learning with TensorBoard
[TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) is a small utility that comes with `TensorFlow` and it will help you inspect your training progress and evaluate the quality of different models. 
To start `tensorboard`, you need to point it to a folder which contains the training logs of a model.

To visualize only one model (experiment), run:

```
tensorboard --logdir=data/models/<model_name>
```

To visualize multiple models (experiments) at the same time, run:

```
tensorboard --logdir=data/models
```

The second command will automatically pick up the training logs of all models that are to be found under `data/models`.
