import set_seed
import config
import os
from typing import Sequence, Tuple
import pandas as pd

import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import input_pipeline
import models

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

config_ = config.Config()

def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(config_.train['mlflow_experiment_name'])
    with mlflow.start_run():
        mlflow.log_params({
            "learning_rate": config_.train['learning_rate'],
            "regularization": config_.train['regularization'],
            "embedding_dim": config_.train['embedding_dim'],
            "batch_size": config_.train['batch_size'],
            "max_steps": config_.train['max_steps']
        })

        train_meta = pd.read_csv(config_.data['metatrain_path'])
        test_meta = pd.read_csv(config_.data['metatest_path'])
        num_train = len(train_meta)
        num_test = len(test_meta)

        train_ds = input_pipeline.create_dataset(train_meta).repeat().batch(config_.train['batch_size']).prefetch(tf.data.AUTOTUNE)
        test_ds = input_pipeline.create_dataset(test_meta).repeat().batch(config_.train['batch_size'])

        model = models.STLModel(embedding_dim=config_.train['embedding_dim'])
        optimizer = tf.keras.optimizers.Adam(learning_rate=config_.train['learning_rate'])

        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        manager = tf.train.CheckpointManager(checkpoint, config_.train['work_dir'], max_to_keep=3)

        if config_.train['restore_checkpoint']:
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
                loss = (triplet_loss + config_.train['regularization'] * reg_loss) / config_.train['batch_size']
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

        for step in range(config_.train['max_steps']):
            batch = next(train_iter)
            scene, pos_product, neg_product = batch[0], batch[1], batch[2]
            loss = train_step(scene, pos_product, neg_product)

            if step % config_.train['checkpoint_every_steps'] == 0:
                save_path = manager.save(checkpoint_number = step + 1)
                print(f"Saved checkpoint for step {step}: {save_path}")

            if step % config_.train['eval_every_steps'] == 0 and step > 0:
                eval_losses = []
                for _ in range(num_test // config_.train['batch_size']):
                    ebatch = next(test_iter)
                    escene, epos_product, eneg_product = ebatch[0], ebatch[1], ebatch[2]
                    eval_loss = eval_step(escene, epos_product, eneg_product)
                    eval_losses.append(eval_loss)
                eval_loss = tf.reduce_mean(eval_losses) / config_.train['batch_size']
                mlflow.log_metric("eval_loss", eval_loss.numpy(), step=step)

            if step % config_.train['log_every_steps'] == 0 and step > 0:
                mlflow.log_metric("train_loss", loss.numpy(), step=step)
                print(f"Step {step}: train_loss = {loss.numpy()}")

        print(f"Saving model as {config_.train['model_name']}")
        model_path = os.path.join(config_.train['work_dir'], config_.train['model_name'])
        model.save(model_path)
        mlflow.tensorflow.log_model(model, config_.train['model_name'], custom_objects={'STLModel': models.STLModel})

if __name__ == "__main__":
    main()
