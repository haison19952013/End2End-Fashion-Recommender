import pin_util
import input_pipeline
import numpy as np
import config




scene_product = pin_util.get_valid_scene_product(args.image_dir, args.input_file)
print(f"Found {len(scene_product)} valid scene product pairs.")

train, test = input_pipeline.generate_triplets(scene_product, args.num_neg)
num_train = len(train)
num_test = len(test)
print(f"Train triplets: {num_train}")
print(f"Test triplets: {num_test}")

np.random.shuffle(train)
np.random.shuffle(test)
train = np.array(train)
test = np.array(test)

train_ds = input_pipeline.create_dataset(train).repeat().batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = input_pipeline.create_dataset(test).repeat().batch(args.batch_size)