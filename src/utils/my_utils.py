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
  Utilities for handling pinterest images.
"""

from typing import Sequence, Tuple
import os
import json
import tensorflow as tf
from src.data_pipeline import serve_data 

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

def generate_embeddings(unique_items, embed_fn, batch_size, item_type):
    """Generate embeddings for scenes or products."""
    ds = tf.data.Dataset.from_tensor_slices(unique_items).map(serve_data.process_image_with_id)
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

def save_results(filename, scene_key, scores_and_indices, index_to_key):
    """
    Save results of a scoring run as an HTML document.

    Args:
      filename: name of file to save as.
      scene_key: Scene key.
      scores_and_indices: A tuple of (scores, indices).
      index_to_key: A dictionary of index to product key.
    """
    scores, indices = scores_and_indices
    with open(filename, "w") as f:
        f.write("<HTML>\n")
        scene_img = local_file_to_pin_url(scene_key)
        f.write(f"Nearest neighbors to {scene_img}<br>\n")
        for i, (score, idx) in enumerate(zip(scores, indices)):
            product_key = index_to_key[idx]
            product_img = local_file_to_pin_url(product_key)
            f.write(f"Rank {i + 1} Score {score:.6f}<br>{product_img}<br>\n")
        f.write("</HTML>\n")

def local_file_to_pin_url(filename):
    """Converts a local filename to a Pinterest URL."""
    key = filename.split("/")[-1]
    key = key.split(".")[0]
    url = key_to_url(key)
    return f'<img src="{url}">'