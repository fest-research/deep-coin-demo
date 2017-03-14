# Coin recognition
---

This is a small demonstration of the capabilities of discriminative models in deep learning. In particular, small 200x200 pixels images are classified as "containing a 50 cent coin" or "not containing a 50 cent coin". The project is meant as a tutorial, illustrating very basic techniques in deep learning (and in particular ConvNets).

#### Dependencies
Check `requirements.txt`. You can install them with:

```
pip install -r requirements.txt
```

#### Model configuration
You can configure basic hyper-parameters of the classifier in `config.yaml`. The configuration file is loaded from the file system every time `main.py` is started.

#### Training
To start training the model, run:

```
python main.py
```

#### Testing
A small online demo was created to show the pre-trained classifier in action. A camera needs to be connected to your machine. Run:

```
python demo.py --modeldir=./data/models/<model_name> --weights=<weights_filename>.hdf5
```

A live feed from the camera will be processed frame by frame by che selected classifier and the frames will be classified as either "containing a coint" or "not containing a coin" on the fly.

#### Data management
In order to train the model, attention should be paid to the structure of the `data` folder:

* `data/positive` should contain 200x200 images with a coin in them. For this, you can extract `positive.tar.gz`.
* `data/negative` should contain 200x200 images that have no coin in them. For this, you can extract `negative.tar.gz`.
* `data/models` will contain the results of all trained models; you do not need to create this directory manually


