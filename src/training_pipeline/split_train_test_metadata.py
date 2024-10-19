import set_seed
import pin_util
import input_pipeline
import numpy as np
import config
import pandas as pd

config_ = config.Config()
scene_product = pin_util.get_valid_scene_product(config_.data['raw_image_path'], config_.data['metadata_path'])
print(f"Found {len(scene_product)} valid scene product pairs.")

train, test = input_pipeline.generate_triplets(scene_product, config_.data['num_negative_samples'])
num_train = len(train)
num_test = len(test)
print(f"Train triplets: {num_train}")
print(f"Test triplets: {num_test}")

np.random.shuffle(train)
np.random.shuffle(test)
train = np.array(train)
test = np.array(test)
df_train = pd.DataFrame(train, columns=['scene', 'pos', 'neg'])
df_test = pd.DataFrame(test, columns=['scene', 'pos', 'neg'])
df_train.to_csv(config_.data['metatrain_path'], index = False)
df_test.to_csv(config_.data['metatest_path'], index = False)
