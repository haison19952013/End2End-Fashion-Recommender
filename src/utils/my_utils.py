#!/usr/bin/env python
# _*_ coding: utf-8 -*-
# 
# 
# Copyright 2022 Hector Yee, Bryan Bischoff
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
  Utilities for multipurpose use.
"""

from typing import Sequence, Tuple
import os
import json
import tensorflow as tf
from src.data_pipeline import serve_data 
import mlflow
import base64
import numpy as np


def key_to_url(key: str)-> str:
    """
    Converts a pinterest hex key into a url.
    """
    prefix = 'http://i.pinimg.com/400x/%s/%s/%s/%s.jpg'
    return prefix % (key[0:2], key[2:4], key[4:6], key)

def id_to_filename(image_dir: str, id: str) -> str:
    filename = os.path.join(
        image_dir,
        id + ".jpg")
    return filename

def is_valid_file(fname):
    return os.path.exists(fname) and os.path.getsize(fname) > 0

def get_valid_scene_product(image_dir:str, input_file: str) -> Sequence[Tuple[str, str]]:
    """
      Reads in the Shop the look json file and returns a pair of scene and matching products.
    """
    scene_product = []
    with open(input_file, "r") as f:
        data = f.readlines()
        for line in data:
            row = json.loads(line)
            scene = id_to_filename(image_dir, row["scene"])
            product = id_to_filename(image_dir, row["product"])
            if is_valid_file(scene) and is_valid_file(product):
                scene_product.append([scene, product])
    return scene_product

def generate_embeddings(unique_items, embed_fn, batch_size, item_type, file_type = 'path'):
    """Generate embeddings for scenes or products."""
    if file_type is 'path':
        ds = tf.data.Dataset.from_tensor_slices(unique_items).map(serve_data.process_image_with_filepath)
    elif file_type is 'byte':
        ds = tf.data.Dataset.from_tensor_slices(unique_items).map(serve_data.process_image_with_filebyte)
    else:
        raise ValueError("file_type must be 'path' or 'byte'")
    ds = ds.batch(batch_size, drop_remainder=False)
    it = ds.as_numpy_iterator()

    embeddings = {}
    count = 0
    for id_batch, image_batch in it:
        count += 1
        if count % 100 == 0:
            print(f"Created {count * batch_size} {item_type} embeddings.")
        
        result = embed_fn(image_batch)
        for i in range(len(id_batch)):
            current_id = id_batch[i].decode("utf-8")
            embeddings[current_id] = result[i].numpy().tolist()

    return embeddings

def save_embeddings(embeddings, filename):
    """Save embeddings as a JSON file."""
    with open(filename, "w") as f:
        json.dump(embeddings, f)
    print(f"Embeddings saved to {filename}")

def find_top_k(scene_embedding, product_embeddings, k):
    """
    Finds the top K nearest product embeddings to the scene embedding.
    
    Args:
      scene_embedding: embedding vector for the scene
      product_embedding: embedding vectors for the products.
      k: number of top results to return.
    """
    scores = tf.reduce_sum(scene_embedding * product_embeddings, axis=-1)
    result = tf.nn.top_k(scores, k=k)
    top_k_values, top_k_indices = result.values.numpy(), result.indices.numpy()
    return top_k_values, top_k_indices

def export_recommendation(scene_bytes, mime_type, scores_and_indices, index_to_key, to_html=True, filename=None):
    """
    Save or return results of a scoring run as an HTML document.

    Args:
      scene_bytes: Byte content of the scene (image data).
      mime_type: MIME type of the image (e.g., 'image/jpeg' or 'image/png').
      scores_and_indices: A tuple of (scores, indices).
      index_to_key: A dictionary mapping index to product key.
      save: Whether to save the HTML to a file (default is True).
      filename: Name of the file to save as (only required if save=True).
    """
    # If saving is requested, ensure filename is provided
    if to_html and not filename:
        raise ValueError("Filename must be provided if save=True")

    scores, indices = scores_and_indices

    # Start generating the HTML content
    html_content = "<HTML>\n"
    
    
    # Create the HTML for the scene image
    scene_img_html_src = map_user_file_to_html_source(scene_bytes, mime_type)
    html_content += f"Nearest neighbors to {scene_img_html_src}<br>\n"
    img_html_dict = {"scene": scene_img_html_src, 'recommendation':[], 'score':[]}
    
    # Generate the HTML for the recommendations
    for i, (score, idx) in enumerate(zip(scores, indices)):
        product_key = index_to_key[idx]
        product_img_html_src = map_pin_file_to_html_source(product_key)  # Assuming this function exists
        html_content += f"Rank {i + 1} Score {score:.6f}<br>{product_img_html_src}<br>\n"
        img_html_dict['recommendation'].append(product_img_html_src)
        img_html_dict['score'].append(score)
    
    html_content += "</HTML>\n"

    # If save is True, write the content to the specified file
    if to_html:
        with open(filename, "w") as f:
            f.write(html_content)
    else:
        # If save is False, return the HTML content as a string
        return img_html_dict



def map_pin_file_to_html_source(filename):
    """Converts a local filename to a Pinterest URL."""
    key = filename.split("/")[-1]
    key = key.split(".")[0]
    url = key_to_url(key)
    return f'<img src="{url}">'

def map_user_file_to_html_source(file_bytes, mime_type=None):
    """Converts a local image file to a base64-encoded image for HTML embedding."""
    
    # Encode the file bytes into Base64
    encoded_image = base64.b64encode(file_bytes).decode("utf-8")
    
    # Ensure we have a valid MIME type (default to image/jpeg if unknown)
    if mime_type is None:
        mime_type = "image/jpeg"
    
    # Return the image embedded as a data URL in HTML
    return f'<img src="data:{mime_type};base64,{encoded_image}">'


def load_registered_model(model_name, tag = 'champion'):
    model_uri = f"models:/{model_name}@{tag}"
    loaded_model = mlflow.tensorflow.load_model(model_uri)
    return loaded_model

def load_product_embedding(product_embed_path):
    with open(product_embed_path, "r") as f:
        product_dict = json.load(f)
    # Map product embeddings and scene embeddings to NumPy arrays
    index_to_key = {}
    product_embeddings = []
    for index, (key, vec) in enumerate(product_dict.items()):
        index_to_key[index] = key
        product_embeddings.append(np.array(vec))
    
    product_embeddings = np.stack(product_embeddings, axis=0)
    return product_embeddings, index_to_key