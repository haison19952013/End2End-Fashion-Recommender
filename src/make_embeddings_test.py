
"""
  Generates embedding files given a model and a catalog.
"""
import config
import random
import json
import os
import argparse
import numpy as np
import tensorflow as tf
from typing import Sequence, Tuple
import models  # Your custom model library
import input_pipeline  # Your input pipeline library
import pin_util  # Utility for getting the valid scene-product pairs
import mlflow.tensorflow

config_ = config.Config()

# load the model from the Model Registry and score
model_name = 'recsys-fashion-model'
# model_uri = f"models:/{config_.train['model_name']}/1" 
model_uri = f"models:/{config_.train['model_name']}@{'champion'}"
loaded_model = mlflow.tensorflow.load_model(model_uri)
print("--")

# experiment_id ='216418323594056424'
# run_id = '0e397d7cc4d641b4b99b9b346af4940a'
# artifact_name = 'pinterest_stl_model_rc1'
# # Path to the model artifact in mlruns
# model_path = f"../mlruns/{experiment_id}/{run_id}/artifacts/{artifact_name}"

# # Load the TensorFlow model
# loaded_model = mlflow.tensorflow.load_model(model_path)

# get_scene_embed = loaded_model.get_scene_embed
# get_product_embed = loaded_model.get_product_embed


# scene_product = pin_util.get_valid_scene_product(config_.data['raw_image_path'], config_.data['metadata_path'])
# print(f"Found {len(scene_product)} valid scene product pairs.")
# # Extract unique scenes and products
# unique_scenes = np.array(list(set(x[0] for x in scene_product)))
# unique_products = np.array(list(set(x[1] for x in scene_product)))
# print(f"Found {len(unique_scenes)} unique scenes.")
# print(f"Found {len(unique_products)} unique products.")


# def generate_embeddings(unique_items, embed_fn, input_pipeline, batch_size, item_type):
#     """Generate embeddings for scenes or products."""
#     ds = tf.data.Dataset.from_tensor_slices(unique_items).map(input_pipeline.process_image_with_id)
#     ds = ds.batch(batch_size, drop_remainder=True)
#     it = ds.as_numpy_iterator()

#     embeddings = {}
#     count = 0
#     for id, image in it:
#         count += 1
#         if count % 100 == 0:
#             print(f"Created {count * batch_size} {item_type} embeddings.")
        
#         result = embed_fn(image)
#         for i in range(batch_size):
#             current_id = id[i].decode("utf-8")
#             embeddings[current_id] = result[i].numpy().tolist()

#     return embeddings

# def save_embeddings(embeddings, filename):
#     """Save embeddings as a JSON file."""
#     with open(filename, "w") as f:
#         json.dump(embeddings, f)
#     print(f"Embeddings saved to {filename}")

# # Process scenes and generate embeddings
# scene_dict = generate_embeddings(unique_scenes, get_scene_embed, input_pipeline, 16, "scene")
# save_embeddings(scene_dict, os.path.join('data', "scene_embed.json"))

# # Process products and generate embeddings
# product_dict = generate_embeddings(unique_products, get_product_embed, input_pipeline, 16, "product")
# save_embeddings(product_dict, os.path.join('data', "product_embed.json"))