import tensorflow as tf
try:
    from . import pin_util  # Relative import for package context (e.g., pytest)
    from . import input_pipeline
    from . import models
except ImportError:
    import pin_util  # Fallback for running the script directly
    import input_pipeline
    import models
from typing import Sequence, Tuple
import numpy as np


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

scene_product = pin_util.get_valid_scene_product('F:\database\shop_the_look-v1', '../STL-Dataset/fashion.json')
print(f"Found {len(scene_product)} valid scene product pairs.")

train, test = generate_triplets(scene_product, 5)
num_train = len(train)
num_test = len(test)
print(f"Train triplets: {num_train}")
print(f"Test triplets: {num_test}")

train = np.array(train)
test = np.array(test)

train_ds = input_pipeline.create_dataset(train).repeat().batch(16).prefetch(tf.data.AUTOTUNE)
test_ds = input_pipeline.create_dataset(test).repeat().batch(16)

model = models.STLModel(output_size=64)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
@tf.function
def train_step(scene, pos_product, neg_product):
    with tf.GradientTape() as tape:
        pos_dist, neg_dist, scene_embed, pos_embed, neg_embed = model(scene, pos_product, neg_product)
        triplet_loss = tf.reduce_sum(tf.nn.relu(1.0 + neg_dist - pos_dist))
        reg_loss = tf.reduce_sum(tf.nn.relu(tf.sqrt(tf.reduce_sum(tf.square(scene_embed), axis=-1)) - 1.0)) + \
                    tf.reduce_sum(tf.nn.relu(tf.sqrt(tf.reduce_sum(tf.square(pos_embed), axis=-1)) - 1.0)) + \
                    tf.reduce_sum(tf.nn.relu(tf.sqrt(tf.reduce_sum(tf.square(neg_embed), axis=-1)) - 1.0))
        loss = (triplet_loss + 0.2 * reg_loss) / 16.
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
train_iter = iter(train_ds)
test_iter = iter(test_ds)

for step in range(10):
    batch = next(train_iter)
    print(batch[0].shape)
    scene, pos_product, neg_product = batch[0], batch[1], batch[2]
    loss = train_step(scene, pos_product, neg_product)
    print(loss.numpy())

