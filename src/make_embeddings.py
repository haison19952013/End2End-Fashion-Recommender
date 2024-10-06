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
  Generates embedding files given a model and a catalog.
"""

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

# Argument parser for CLI arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Generates embeddings for scenes and products.")
    parser.add_argument('--input_file', required=True, type=str, help="Input catalog JSON file.")
    parser.add_argument('--image_dir', required=True, type=str, help="Directory containing downloaded images.")
    parser.add_argument('--out_dir', default="/tmp", type=str, help="Output directory for embeddings.")
    parser.add_argument('--output_size', default=64, type=int, help="Size of the output embeddings.")
    parser.add_argument('--model_name', required=True, type=str, help="Pre-trained model name for embedding generation.")
    parser.add_argument('--batch_size', default=8, type=int, help="Batch size for embedding generation.")
    
    return parser.parse_args()

def generate_embeddings(unique_items, embed_fn, input_pipeline, batch_size, item_type):
    """Generate embeddings for scenes or products."""
    ds = tf.data.Dataset.from_tensor_slices(unique_items).map(input_pipeline.process_image_with_id)
    ds = ds.batch(batch_size, drop_remainder=True)
    it = ds.as_numpy_iterator()

    embeddings = {}
    count = 0
    for id, image in it:
        count += 1
        if count % 100 == 0:
            print(f"Created {count * batch_size} {item_type} embeddings.")
        
        result = embed_fn(image)
        for i in range(batch_size):
            current_id = id[i].decode("utf-8")
            embeddings[current_id] = result[i].tolist()

    return embeddings

def save_embeddings(embeddings, filename):
    """Save embeddings as a JSON file."""
    with open(filename, "w") as f:
        json.dump(embeddings, f)
    print(f"Embeddings saved to {filename}")

def main():
    # Parse command-line arguments
    args = parse_args()

    # Load valid scene-product pairs
    scene_product = pin_util.get_valid_scene_product(args.image_dir, args.input_file)
    print(f"Found {len(scene_product)} valid scene product pairs.")

    # Extract unique scenes and products
    unique_scenes = np.array(list(set(x[0] for x in scene_product)))
    unique_products = np.array(list(set(x[1] for x in scene_product)))
    print(f"Found {len(unique_scenes)} unique scenes.")
    print(f"Found {len(unique_products)} unique products.")

    # Load the model and its parameters
    model = models.STLModel(embedding_dim=args.output_size)
    print(f"Loading model from {args.model_name}")
    
    # Load model state
    with open(args.model_name, "rb") as f:
        data = f.read()
        state = model.restore_state(data)
    assert state is not None, "Failed to load the model state."

    # Define functions to get scene and product embeddings using the model
    def get_scene_embed(x):
        return model.apply(state['params'], x, method=models.STLModel.get_scene_embed)

    def get_product_embed(x):
        return model.apply(state['params'], x, method=models.STLModel.get_product_embed)

    # Process scenes and generate embeddings
    scene_dict = generate_embeddings(unique_scenes, get_scene_embed, input_pipeline, args.batch_size, "scene")
    save_embeddings(scene_dict, os.path.join(args.out_dir, "scene_embed.json"))

    # Process products and generate embeddings
    product_dict = generate_embeddings(unique_products, get_product_embed, input_pipeline, args.batch_size, "product")
    save_embeddings(product_dict, os.path.join(args.out_dir, "product_embed.json"))



if __name__ == "__main__":
    main()
