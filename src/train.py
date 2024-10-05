import argparse
import json
import os
from typing import Sequence, Tuple

import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow

# Assuming these modules exist and are compatible with TensorFlow
try:
    from . import input_pipeline
    from . import models
    from . import pin_util
except ImportError:
    import pin_util  # Fallback for running the script directly
    import input_pipeline
    import models


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model for the shop the look dataset.")
    parser.add_argument("--input_file", default="STL-Dataset/fashion.json", help="Input cat json file.")
    parser.add_argument("--image_dir", default="artifacts/shop_the_look:v1", help="Directory containing downloaded images.")
    parser.add_argument("--num_neg", type=int, default=5, help="How many negatives per positive.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--regularization", type=float, default=0.1, help="Regularization.")
    parser.add_argument("--output_size", type=int, default=32, help="Size of output embedding.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--log_every_steps", type=int, default=100, help="Log every this step.")
    parser.add_argument("--eval_every_steps", type=int, default=2000, help="Eval every this step.")
    parser.add_argument("--checkpoint_every_steps", type=int, default=100000, help="Checkpoint every this step.")
    parser.add_argument("--max_steps", type=int, default=30000, help="Max number of steps.")
    parser.add_argument("--work_dir", default="tmp", help="Work directory.")
    parser.add_argument("--model_name", default="pinterest_stl_model", help="Model name.")
    parser.add_argument("--restore_checkpoint", action="store_true", help="If true, restore checkpoint.")
    parser.add_argument("--mlflow_experiment_name", default="recsys-pinterest", help="MLflow experiment name.")
    return parser.parse_args()

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

def main():
    args = parse_args()
    
    mlflow.set_experiment(args.mlflow_experiment_name)
    with mlflow.start_run():
        mlflow.log_params({
            "learning_rate": args.learning_rate,
            "regularization": args.regularization,
            "output_size": args.output_size,
            "batch_size": args.batch_size,
            "max_steps": args.max_steps
        })

        scene_product = pin_util.get_valid_scene_product(args.image_dir, args.input_file)
        print(f"Found {len(scene_product)} valid scene product pairs.")

        train, test = generate_triplets(scene_product, args.num_neg)
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

        model = models.STLModel(output_size=args.output_size)
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        manager = tf.train.CheckpointManager(checkpoint, args.work_dir, max_to_keep=3)

        if args.restore_checkpoint:
            checkpoint.restore(manager.latest_checkpoint)
            print(f"Restored from {manager.latest_checkpoint}")

        @tf.function
        def train_step(scene, pos_product, neg_product):
            with tf.GradientTape() as tape:
                pos_dist, neg_dist, scene_embed, pos_embed, neg_embed = model(scene, pos_product, neg_product)
                triplet_loss = tf.reduce_sum(tf.nn.relu(1.0 + neg_dist - pos_dist))
                reg_loss = tf.reduce_sum(tf.nn.relu(tf.sqrt(tf.reduce_sum(tf.square(scene_embed), axis=-1)) - 1.0)) + \
                           tf.reduce_sum(tf.nn.relu(tf.sqrt(tf.reduce_sum(tf.square(pos_embed), axis=-1)) - 1.0)) + \
                           tf.reduce_sum(tf.nn.relu(tf.sqrt(tf.reduce_sum(tf.square(neg_embed), axis=-1)) - 1.0))
                loss = (triplet_loss + args.regularization * reg_loss) / args.batch_size
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        @tf.function
        def eval_step(scene, pos_product, neg_product):
            pos_dist, neg_dist, _, _, _ = model(scene, pos_product, neg_product)
            loss = tf.reduce_sum(tf.nn.relu(1.0 + neg_dist - pos_dist))
            return loss

        train_iter = iter(train_ds)
        test_iter = iter(test_ds)

        for step in range(args.max_steps):
            batch = next(train_iter)
            scene, pos_product, neg_product = batch[0], batch[1], batch[2]
            loss = train_step(scene, pos_product, neg_product)

            if step % args.checkpoint_every_steps == 0:
                save_path = manager.save(checkpoint_number = step + 1)
                print(f"Saved checkpoint for step {step}: {save_path}")

            if step % args.eval_every_steps == 0 and step > 0:
                eval_losses = []
                for _ in range(num_test // args.batch_size):
                    ebatch = next(test_iter)
                    escene, epos_product, eneg_product = ebatch[0], ebatch[1], ebatch[2]
                    eval_loss = eval_step(escene, epos_product, eneg_product)
                    eval_losses.append(eval_loss)
                eval_loss = tf.reduce_mean(eval_losses) / args.batch_size
                mlflow.log_metric("eval_loss", eval_loss.numpy(), step=step)

            if step % args.log_every_steps == 0 and step > 0:
                mlflow.log_metric("train_loss", loss.numpy(), step=step)
                print(f"Step {step}: train_loss = {loss.numpy()}")

        print(f"Saving model as {args.model_name}")
        model_path = os.path.join(args.work_dir, args.model_name)
        model.save(model_path)
        mlflow.tensorflow.log_model(model, args.model_name, custom_objects={'STLModel': models.STLModel})

if __name__ == "__main__":
    main()
