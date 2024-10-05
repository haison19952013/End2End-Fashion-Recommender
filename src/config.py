class Config():
    def __init__(self):
        self.data = {'raw_image_path': '../data/shop_the_look-v1',
                     'train_path': '../data/train.tfds',
                    'test_path': '../data/test.tfds'}
        self.train = {'learning_rate': 1e-3,
                      'regularization': 1e-4,
                      'embedding_dim':64,
                      'batch_size': 32,
                      'log_every_steps': 100,
                      'eval_every_steps': 1000,
                      'checkpoint_every_steps': 10000,
                      'max_steps': 20,
                      'work_dir': 'tmp',
                      'model_name':'fashion_recommender',
                      'restore_checkpoint': False,
                      'mlflow_experiment_name': 'recsys-pinterest'}
                      