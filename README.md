## Build Docker image for API server
### Build

```bash
docker build -t haison19952013/fashion_recsys:v0.0.0 -f src/api_service/Dockerfile .
```
### Run
```bash
docker run -d --name my_app -p 8000:8000 haison19952013/fashion_recsys:v0.0.0
```

### Debug
```bash
docker run -it --name my_debug_container haison19952013/fashion_recsys:v0.0.0 /bin/bash
```

## Build Docker image for Web apps
### Build

```bash
docker build -t haison19952013/fashion_recsys_web:v0.0.0 -f src/web_service/Dockerfile .
```
### Run
```bash
docker run -d --name my_app -p 8051:8051 haison19952013/fashion_recsys_web:v0.0.0
```

### Debug
```bash
docker run -it --name my_debug_container haison19952013/fashion_recsys_web:v0.0.0 /bin/bash
```

## Run Docker compose to start both API server and Web apps
```bash
docker compose -f api-web-docker-compose.yaml up -d 
```

## Test Helm chart to deploy the whole system in minikube using NodePort
### Build
```bash
kubectl create ns rec-sys-serving
kubens rec-sys-serving
cd helm-charts/rec-sys-services
helm upgrade --install rec-sys-services .
```
### Get my-web-service URL
```bash
minikube service my-web-service --url -n rec-sys-serving
```


Installation instructions
=========================

This tutorial has been tested on Ubuntu (under Windows, WSL2 Ubuntu).

OS level installs
=================
Some packages require python 3.9 so please ensure that you are not using python 3.10

To get python 3.9 and the virtual environment on Ubuntu

sudo apt-install python3.9
sudo apt-install python3.9-venv

Python virtual environment setup
================================

To setup the python virutal environment, in the parent directory above this one run:

python3 -m venv venv_name

cd venv_name

source bin/activate

Then go to a code directory e.g.

cd pinterest

pip install -r requirements.txt

If you want the GPU version of jax please follow the latest instructions on the jax webpage.

If you wish to skip all the data processing steps and skip straight to training embeddings you can make use of the pre-made weights and biases artifacts.
Skip to the train word embeddings section.

Getting the Data
================

For this data set we will be using the Shop The Look Data set from

[Shop the Look data set](https://github.com/kang205/STL-Dataset)

[Wang-Cheng Kang](http://kwc-oliver.com), Eric Kim, [Jure Leskovec](https://cs.stanford.edu/people/jure/), Charles Rosenberg, [Julian McAuley](http://cseweb.ucsd.edu/~jmcauley/) (2019). *[Complete the Look: Scene-based Complementary Product Recommendation.](https://arxiv.org/pdf/1812.01748.pdf)* In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR'19)

A copy of the STL dataset is in the directory STL-Dataset.
The originals can be fetched directly a the github repo above.

Running sample code
===================

To run a random item recommender:

python3 random_item_recommender.py --input_file=STL-Dataset/fashion-cat.json --output_html=output.html

To view the output open output.html with your favorite browser

e.g. google-chrome output.html

To fetch the images from pinterest for training, this will take a while!
Pinterest might also block you for scraping if you don't set the sleep timeout to a high enough value.
=======
Sometimes pinterest might block the download by rate limiting,
so the code has been written to sleep if pinterest blocks the traffic in order to get
under the rate limit. If that happens the code will retry by adding a second to the sleep time
each try.

python3 fetch_images.py --input_file=STL-Dataset/fashion.json --output_dir=images/ --max_lines=100000 --sleep_time=5

Alternatively if you have a weights and biases account:
=======
They were uploaded to wandb as an artifact using

wandb artifact put -n "recsys-pinterest/shop_the_look" -d "Images from shop the look" -t "images" images

 wandb artifact get building-recsys/recsys-pinterest/shop_the_look:latest

The images will then be in

./artifacts/shop_the_look:v1

Training the Model
==================

python3 train_shop_the_look.py --input_file=STL-Dataset/fashion.json --image_dir=./artifacts/shop_the_look\:v1 --max_steps=30000 --learning_rate=0.0001 --regularization=0.2  --output_size=64 --checkpoint_every_steps=10000 --restore_checkpoint=True --model_name=pinterest_stl_model_rc1

A pre-trained model has been uploaded and can be fetched using

wandb artifact get building-recsys/recsys-pinterest/pinterest_stl_model_rc1:v0

Hyperparameter tuning
=====================

Create a sweep

wandb sweep --project recsys-pinterest sweep.yaml

Start the sweep agent using the command line printed after the above command.


Generating the embedding database
=================================

 python3 make_embeddings.py --input_file=STL-Dataset/fashion.json --image_dir=./artifacts/shop_the_look\:v1 --model_name=./artifacts/pinterest_stl_model_rc1\:v0/pinterest_stl.model --output_size=64

 A copy of the embeddings can be optained using

 wandb artifact get building-recsys/recsys-pinterest/scene_product_embeddings:v0

 Make recommendations
 ====================

 python3 make_recommendations.py --product_embed=./artifacts/scene_product_embeddings\:v0/product_embed.json --scene_embed=./artifacts/scene_product_embeddings\:v0/scene_embed.json

 Sample results can be seen at

 wandb artifact get building-recsys/recsys-pinterest/scene_product_results:v0

 # Shop the Look Dataset

This repository contains the Pinterest shop the look data used in the following paper:

[Wang-Cheng Kang](http://kwc-oliver.com), Eric Kim, [Jure Leskovec](https://cs.stanford.edu/people/jure/), Charles Rosenberg, [Julian McAuley](http://cseweb.ucsd.edu/~jmcauley/) (2019). *[Complete the Look: Scene-based Complementary Product Recommendation.](https://arxiv.org/pdf/1812.01748.pdf)* In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR'19)

Please cite our paper if you use the datasets.


### Datasets

We provide the two datasets for `fashion` and `home` respectively. Each dataset contains the scene-product pairs in the following format:

```json
Example (fashion.json):
{
    "product": "0027e30879ce3d87f82f699f148bff7e", 
    "scene": "cdab9160072dd1800038227960ff6467", 
    "bbox": [
        0.434097, 
        0.859363, 
        0.560254, 
        1.0
    ]
}
```

The scene and product images are represented by a signature, and the corresponding image URL can be obtained via:

```python
def convert_to_url(signature):
    prefix = 'http://i.pinimg.com/400x/%s/%s/%s/%s.jpg'
    return prefix % (signature[0:2], signature[2:4], signature[4:6], signature)
```   

Also, product category is also provided (e.g. fashion-cat.json). Please refer to the paper for more statistics and details. You could contact me (wckang#ucsd.edu) if there is any question. 

### Bounding Box

The bounding box is represented by four numbers: `(left, top, right, bottom)`. All the numbers are normalized. The coordinates of the top left corner is `(0,0)`.

### TODO
- Add an illustrative example.
