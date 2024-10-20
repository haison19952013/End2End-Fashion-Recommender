class Config():
    def __init__(self):
        self.seed = 42
        self.data = {'metadata_path': 'data/fashion.json',
                    'raw_image_path': 'data/fashion_images_v0',
                    'num_negative_samples': 5,
                    'metatrain_path': 'data/meta_train.csv',
                    'metatest_path': 'data/meta_test.csv',
                    'scene_embed_path': 'data/scene_embed.json',
                    'product_embed_path': 'data/product_embed.json'
                    }
        self.train = {'learning_rate': 0.0001618, # Optimal:  0.0001618
                      'regularization': 0.2076, # Optimal:  0.2076
                      'embedding_dim':64, # Optimal:  64
                      'batch_size': 32,
                      'log_every_steps': 100,
                      'eval_every_steps': 100,
                      'checkpoint_every_steps': 5000,
                      'max_steps': 30000,
                      'work_dir': './tmp',
                      'model_name':'recsys-fashion-model',
                      'restore_checkpoint': True,
                      'mlflow_experiment_name': 'recsys-fashion-experiment'}
                      