
"""
  Generates embedding files given a model and a catalog.
"""
import numpy as np
from src import config
from src.utils import my_utils  # Utility for getting the valid scene-product pairs

def main():
  config_ = config.Config()

  # load the model from the Model Registry and score
  loaded_model = my_utils.load_registered_model(model_name = config_.train['model_name'], tag = 'champion')
  print("--")
  
  get_scene_embed = loaded_model.get_scene_embed
  get_product_embed = loaded_model.get_product_embed

  scene_product = my_utils.get_valid_scene_product(config_.data['raw_image_path'], config_.data['metadata_path'])
  print(f"Found {len(scene_product)} valid scene product pairs.")
  # Extract unique scenes and products
  unique_scenes = np.array(list(set(x[0] for x in scene_product)))
  unique_products = np.array(list(set(x[1] for x in scene_product)))
  print(f"Found {len(unique_scenes)} unique scenes.")
  print(f"Found {len(unique_products)} unique products.")

  # # Process scenes and generate embeddings
  scene_dict = my_utils.generate_embeddings(unique_scenes, get_scene_embed, 16, "scene")
  my_utils.save_embeddings(scene_dict, config_.data['scene_embed_path'])

  # # #  Process products and generate embeddings
  product_dict = my_utils.generate_embeddings(unique_products, get_product_embed, 16, "product")
  my_utils.save_embeddings(product_dict, config_.data['product_embed_path'])

if __name__ == '__main__':
  main()