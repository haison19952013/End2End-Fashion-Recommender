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

from typing import Sequence, Tuple, Set

import numpy as np
import tensorflow as tf

def generate_triplets(
    scene_product: Sequence[Tuple[str, str]],
    num_neg: int) -> Sequence[Tuple[str, str, str]]:
    """Generate positive and negative triplets."""
    count = len(scene_product)
    train = []
    test = []
    for i in range(count):
        scene, pos = scene_product[i]
        is_test = i % 10 == 0
        neg_indices = np.random.randint(0, count - 1, num_neg)
        for neg_idx in neg_indices:
            _, neg = scene_product[neg_idx]
            if is_test:
                test.append((scene, pos, neg))
            else:
                train.append((scene, pos, neg))
    return train, test

def normalize_image(img):
  img = tf.cast(img, dtype=tf.float32)
  img = (img / 255.0) - 0.5
  return img

def process_image(file_path):
  x = tf.io.read_file(file_path)
  # x = tf.io.decode_jpeg(x, channels=3)
  x = tf.io.decode_image(x, channels=3)
  x = tf.image.resize_with_crop_or_pad(x, 512, 512)
  x = normalize_image(x)
  return x

def process_image_with_id(id):
  image = process_image(id)
  return id, image

def process_triplet(x):
  x = (process_image(x[0]), process_image(x[1]), process_image(x[2]))
  return x

def create_dataset(
    triplet: Sequence[Tuple[str, str, str]]):
    """Creates a triplet dataset.

    Args:
      triplet: filenames of scene, positive product, negative product.
    """
    ds = tf.data.Dataset.from_tensor_slices(triplet)
    ds = ds.map(process_triplet)
    return ds