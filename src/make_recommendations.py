#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
  Given embedding files makes recommendations.
"""

import json
import os
import argparse
import numpy as np
import tensorflow as tf
import pin_util

# Define command-line arguments using argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Given embedding files makes recommendations")
    
    parser.add_argument('--product_embed', type=str, required=True, help='Product embedding JSON file')
    parser.add_argument('--scene_embed', type=str, required=True, help='Scene embedding JSON file')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top-scoring products to return per scene')
    parser.add_argument('--output_dir', type=str, default='/tmp', help='Location to write output HTML files')
    parser.add_argument('--max_results', type=int, default=100, help='Maximum number of scenes to score')
    
    return parser.parse_args()

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

def local_file_to_pin_url(filename):
    """Converts a local filename to a Pinterest URL."""
    key = filename.split("/")[-1]
    key = key.split(".")[0]
    url = pin_util.key_to_url(key)
    return f'<img src="{url}">'

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

def main():
    # Parse command-line arguments
    args = parse_args()
    
    tf.config.set_visible_devices([], 'GPU')  # Disable GPU
    tf.compat.v1.enable_eager_execution()

    # Load product and scene embeddings from JSON files
    with open(args.product_embed, "r") as f:
        product_dict = json.load(f)
    with open(args.scene_embed, "r") as f:
        scene_dict = json.load(f)

    # Map product embeddings and scene embeddings to NumPy arrays
    index_to_key = {}
    product_embeddings = []
    for index, (key, vec) in enumerate(product_dict.items()):
        index_to_key[index] = key
        product_embeddings.append(np.array(vec))
    
    product_embeddings = np.stack(product_embeddings, axis=0)

    # Iterate over scenes and find top-k products for each
    for index, (scene_key, scene_vec) in enumerate(scene_dict.items()):
        scene_embed = np.expand_dims(np.array(scene_vec), axis=0)
        scores_and_indices = find_top_k(scene_embed, product_embeddings, args.top_k)
        
        # Save results as HTML
        filename = os.path.join(args.output_dir, f"{index:05d}.html")
        save_results(filename, scene_key, scores_and_indices, index_to_key)
        
        if index >= args.max_results:
            break

if __name__ == "__main__":
    main()
