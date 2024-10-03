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

"""TensorFlow Models for the shop the look content recommender."""

import tensorflow as tf
from tensorflow.keras import layers

class CNN(tf.keras.Model):
    """Simple CNN."""
    
    def __init__(self, filters, output_size):
        super(CNN, self).__init__()
        self.filters = filters
        self.output_size = output_size
        self.conv_layers = []
        self.bn_layers = []
        
        # Create convolutional, batch normalization, and pooling layers
        for filter in filters:
            self.conv_layers.append(layers.Conv2D(filter, (3, 3), strides=(2, 2), padding='same'))
            self.bn_layers.append(layers.BatchNormalization())
            self.conv_layers.append(layers.Conv2D(filter, (1, 1), strides=(1, 1)))
            self.bn_layers.append(layers.BatchNormalization())

        # Final dense layer to map to the desired output size
        self.dense = layers.Dense(output_size)

    def call(self, inputs, training=False):
        x = inputs
        for i in range(0, len(self.filters) * 2, 2):
            residual = self.conv_layers[i](x)
            x = self.conv_layers[i](x)
            x = self.bn_layers[i](x, training=training)
            x = tf.nn.swish(x)
            x = self.conv_layers[i+1](x)
            x = self.bn_layers[i+1](x, training=training)
            x = tf.nn.swish(x)
            x = x + residual
            x = tf.nn.avg_pool(x, ksize=3, strides=2, padding="SAME")
        
        # Global average pooling over the spatial dimensions
        x = tf.reduce_mean(x, axis=[1, 2])
        x = self.dense(x)
        return x

class STLModel(tf.keras.Model):
    """Shop the look model that takes in a scene and item and computes a score for them."""
    
    def __init__(self, output_size):
        super(STLModel, self).__init__()
        default_filters = [16, 32, 64, 128]
        
        # Define scene CNN and product CNN using the same architecture
        self.scene_cnn = CNN(filters=default_filters, output_size=output_size)
        self.product_cnn = CNN(filters=default_filters, output_size=output_size)

    def get_scene_embed(self, scene):
        return self.scene_cnn(scene, training=False)

    def get_product_embed(self, product):
        return self.product_cnn(product, training=False)

    def call(self, scene, pos_product, neg_product, training=True):
        # Get embeddings for scene and products
        scene_embed = self.scene_cnn(scene, training=training)
        pos_product_embed = self.product_cnn(pos_product, training=training)
        neg_product_embed = self.product_cnn(neg_product, training=training)
        
        # Compute dot products between scene and positive/negative products
        pos_score = tf.reduce_sum(scene_embed * pos_product_embed, axis=-1)
        neg_score = tf.reduce_sum(scene_embed * neg_product_embed, axis=-1)

        return pos_score, neg_score, scene_embed, pos_product_embed, neg_product_embed
