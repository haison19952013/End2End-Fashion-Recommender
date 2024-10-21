"""
  Given embedding files makes recommendations.
"""

import json
import os
import argparse
import numpy as np
from src.utils import my_utils
from src import config
from src.utils import my_utils
import tensorflow as tf
import mimetypes

config_ = config.Config()
# Define command-line arguments using argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Given embedding files makes recommendations")
    parser.add_argument('--scene_path',default = True ,type=str, help='Image file for scene')
    parser.add_argument('--product_embed',default = config_.data['product_embed_path'] ,type=str, help='Product embedding JSON file')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top-scoring products to return per scene')
    parser.add_argument('--output_dir', type=str, default='./output', help='Location to write output HTML files')
    return parser.parse_args()

def main(args):
    # Parse command-line arguments
    assert os.path.exists(args.scene_path), "Scene image file does not exist"
    
    # load the model from the Model Registry and score
    loaded_model = my_utils.load_registered_model(model_name = config_.train['model_name'], tag = 'champion')
    get_scene_embed = loaded_model.get_scene_embed
    unique_scenes = np.array([args.scene_path])
    scene_dict = my_utils.generate_embeddings(unique_scenes, get_scene_embed, 16, "scene")

    # Map product embeddings and scene embeddings to NumPy arrays
    product_embeddings, index_to_key = my_utils.load_product_embedding(args.product_embed)

    # Iterate over scenes and find top-k products for each
    for index, (scene_path, scene_vec) in enumerate(scene_dict.items()):
        scene_embed = np.expand_dims(np.array(scene_vec), axis=0)
        scores_and_indices = my_utils.find_top_k(scene_embed, product_embeddings, args.top_k)
        
        # Save results as HTML
        filename = os.path.join(args.output_dir, f"{scene_path.split('/')[-1]}.html")
        scene_bytes = tf.io.read_file(args.scene_path).numpy()  # Convert Tensor to bytes
        mime_type, _ = mimetypes.guess_type(args.scene_path)
        my_utils.export_recommendation_to_html(scene_bytes, mime_type, scores_and_indices, index_to_key, save = True, filename = filename)
        
if __name__ == "__main__":
    main(args = parse_args())
