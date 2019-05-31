import os
import pickle
import config
import numpy as np
from tqdm import tqdm


def read_glove(glove_file=config.glove_file, glove_vocab=config.glove_vocab, least_freq=10000, most_freq=None):
    """
    read pre-trained glove
    cost about 3 minutes
    :return: numpy tensor of the embeddings and the vocab
    """
    if most_freq is None:
        save_file_name = f'glove_vocab_{least_freq}~.vcb'
    else:
        save_file_name = f'glove_vocab_{least_freq}~{most_freq}.vcb'
    save_file = os.path.join(config.chinese_ROOT_DIR, save_file_name)
    map_dict = {}
    if save_file is not None and os.path.isfile(save_file):
        emb_glove, vocab = pickle.load(open(save_file, 'rb'))
    else:
        vocab = ['<pad>']  # add padding token
        for line in tqdm(open(glove_vocab), desc='Reading Vocab'):
            line = line.strip('\n').split(' ')
            word = line[0]
            freq = int(line[1])
            if most_freq is None:
                if freq > least_freq:
                    vocab.append(word)
            else:
                if least_freq < freq <= most_freq:
                    vocab.append(word)
        vocab_set = set(vocab)  # hash map to speed up retrieving
        for i, line in enumerate(tqdm(open(glove_file), desc='Reading Glove Emb.')):
            line = line.strip('\n').split(' ')
            if len(line) > 0:
                if line[0] in vocab_set or line[0] == '<unk>':
                    map_dict[line[0]] = np.array(line[1:])
        emb_glove = np.zeros((len(vocab), len(map_dict['<unk>'])), dtype=np.float32)
        for i, word in enumerate(tqdm(vocab, desc='Building Tensor.')):
            try:
                emb_glove[i] = map_dict[word]
            except KeyError:
                continue
        if save_file is not None:
            pickle.dump((emb_glove, vocab), open(save_file, 'wb'))
    return emb_glove, vocab


if __name__ == '__main__':
    read_glove(least_freq=10000)
    read_glove(least_freq=1000, most_freq=10000)
    read_glove(least_freq=100, most_freq=1000)
    read_glove(least_freq=1, most_freq=100)
