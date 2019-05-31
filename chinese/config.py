import os

chinese_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
glove_file = os.path.join(chinese_ROOT_DIR, 'ja_neologd_vectors.txt')
glove_vocab = os.path.join(chinese_ROOT_DIR, 'ja_neologd_vocab.txt')