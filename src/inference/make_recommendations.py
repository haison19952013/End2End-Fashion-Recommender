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
from src.utils import my_utils
from src import config
import mlflow.tensorflow


config_ = config.Config()
# Define command-line arguments using argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Given embedding files makes recommendations")
    parser.add_argument('--scene_image',default = True ,type=str, help='Image file for scene')
    parser.add_argument('--product_embed',default = config_.data['product_embed_path'] ,type=str, help='Product embedding JSON file')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top-scoring products to return per scene')
    parser.add_argument('--output_dir', type=str, default='./output', help='Location to write output HTML files')
    
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # load the model from the Model Registry and score
    model_name = config_.train['model_name']
    model_uri = f"models:/{model_name}@{'champion'}"
    loaded_model = mlflow.tensorflow.load_model(model_uri)
    get_scene_embed = loaded_model.get_scene_embed
    unique_scenes = np.array([args.scene_image])
    scene_dict = my_utils.generate_embeddings(unique_scenes, get_scene_embed, 16, "scene")

    # Load product and scene embeddings from JSON files
    with open(args.product_embed, "r") as f:
        product_dict = json.load(f)

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
        scores_and_indices = my_utils.find_top_k(scene_embed, product_embeddings, args.top_k)
        
        # Save results as HTML
        filename = os.path.join(args.output_dir, f"{scene_key.split('/')[-1]}.html")
        my_utils.save_results(filename, scene_key, scores_and_indices, index_to_key)
        

if __name__ == "__main__":
    main()
