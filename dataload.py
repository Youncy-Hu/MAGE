
from typing import Any, Dict, List
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import os
import json
from PIL import Image, ImageFilter
import lmdb
import pickle
from decord import VideoReader

class LmdbReader(Dataset):
    def __init__(self, lmdb_path: str, shuffle: bool = True, percentage: float = 100):
        self.lmdb_path = lmdb_path
        self.shuffle = shuffle

        assert percentage > 0, "Cannot load dataset with 0 percent original size."
        self.percentage = percentage

        env = lmdb.open(
            self.lmdb_path, subdir=False, readonly=True, lock=False,
            readahead=False, map_size=1099511627776 * 2,
        )
        self.db_txn = env.begin()

        self._keys = [
            f"{i}".encode("ascii") for i in range(env.stat()["entries"])
        ]
        if percentage < 100.0:
            retain_k: int = int(len(self._keys) * percentage / 100.0)
            random.shuffle(self._keys)
            self._keys = self._keys[:retain_k]

        self.shuffle_seed = 0

    def set_shuffle_seed(self, seed: int):
        r"""Set random seed for shuffling data."""
        self.shuffle_seed = seed

    def get_keys(self) -> List[bytes]:
        r"""Return list of keys, useful while saving checkpoint."""
        return self._keys

    def set_keys(self, keys: List[bytes]):
        r"""Set list of keys, useful while loading from checkpoint."""
        self._keys = keys

    def __getstate__(self):
        r"""
        This magic method allows an object of this class to be pickable, useful
        for dataloading with multiple CPU workers. :attr:`db_txn` is not
        pickable, so we remove it from state, and re-instantiate it in
        :meth:`__setstate__`.
        """
        state = self.__dict__
        state["db_txn"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state

        env = lmdb.open(
            self.lmdb_path, subdir=False, readonly=True, lock=False,
            readahead=False, map_size=1099511627776 * 2,
        )
        self.db_txn = env.begin()

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, idx: int):
        datapoint_pickled = self.db_txn.get(self._keys[idx])
        video, caption = pickle.loads(datapoint_pickled)

        return video, caption

class MovingMnistLMDB(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        frames_length: int,
        sample_speed: list,
        image_transform=None,
    ):
        self.reader = LmdbReader(data_root + split + '.lmdb')
        self.transform = image_transform
        self.frames_length = frames_length
        self.sample_speed = sample_speed
        self.vocab = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '0': 3, '1': 4, '2': 5, '3': 6, '4': 7, '5': 8, '6': 9,
                      '7': 10, '8': 11, '9': 12, 'the': 13, 'digit': 14, 'and': 15, 'is': 16, 'are': 17, 'bouncing': 18,
                      'moving': 19, 'here': 20, 'there': 21, 'around': 22, 'jumping': 23, 'up': 24, 'down': 25,
                      'left': 26, 'right': 27, 'then': 28, '.': 29
                      }
        self.padding_idx = self.vocab['[PAD]']

    def __len__(self):
        return len(self.reader)

    def sent2matrix(self, x):
        words = x.split()
        m = np.int32(np.zeros((1, len(words))))

        for i in range(len(words)):
            m[0, i] = self.vocab[words[i]]
        return m

    def tokens2sent(self, tokens):
        vocab_reverse = {}
        for word in self.vocab:
            index = self.vocab[word]
            vocab_reverse[index] = word
        text = ""
        for i in range(tokens.shape[0]):
            text = text + " " + vocab_reverse[tokens[i]]
        return text

    def __getitem__(self, idx):
        images_raw, caption = self.reader[idx]
        caption_tokens = self.sent2matrix(caption)
        caption_tokens = np.insert(caption_tokens, 0, self.vocab['[CLS]'])
        caption_tokens = np.append(caption_tokens, self.vocab['[SEP]'])
        caption_tokens = torch.tensor(caption_tokens, dtype=torch.long)

        frame_num = images_raw.shape[0]
        speed = random.random()
        sample_interval = max(1.0, speed * (self.sample_speed[-1] - self.sample_speed[0]) + self.sample_speed[0])
        choice_idx = np.floor(np.linspace(0, frame_num-1, round(frame_num / sample_interval), endpoint=True)).astype(np.int32)
        images_raw = images_raw[choice_idx]
        images_raw = images_raw[:self.frames_length]
        if self.transform is not None:
            image = self.transform(images_raw.transpose(0, 2, 3, 1)).permute(1, 0, 2, 3)
        else:
            image = images_raw / 255. - 0.5
            image = torch.tensor(image, dtype=torch.float)

        if image.shape[0] < self.frames_length:
            image = torch.cat([image, image[-1].unsqueeze(0).repeat(self.frames_length - image.shape[0], 1, 1, 1)], dim=0)

        return image, caption_tokens, torch.tensor(speed, dtype=torch.float), torch.tensor(len(caption_tokens), dtype=torch.long)

    def collate_fn(self, data):
        caption_tokens = torch.nn.utils.rnn.pad_sequence(
            [d[1] for d in data],
            batch_first=True,
            padding_value=self.padding_idx,
        )

        return torch.stack([d[0] for d in data], dim=0), caption_tokens, torch.stack([d[2] for d in data], dim=0), torch.stack([d[3] for d in data], dim=0)

import nltk
nltk.download('punkt')
class CaterGEN(Dataset):
    def __init__(
        self,
        dataset: str,
        data_root: str,
        split: str,
        frames_length: int,
        sample_speed: list,
        image_transform=None,
        randomness=False
    ):
        mode = 'ambiguous' if randomness else 'explicit'
        with open(os.path.join(data_root, f'{split}_{mode}.json'), 'r') as fp:
            self.anno = json.load(fp)

        self.data_root = data_root
        self.transform = image_transform
        self.frames_length = frames_length
        self.sample_speed = sample_speed
        self.randomness = randomness
        if dataset == 'cater-gen-v1':
            self.vocab = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, 'the': 3, 'cone': 4, 'snitch': 5, 'is': 6,
                          'sliding': 7, 'picked': 8, 'placed': 9, 'containing': 10, 'rotating': 11, 'and': 12, 'to': 13,
                          'up': 14, '(': 15, ')': 16, '1': 17, '2': 18, '3': 19, '-1': 20, '-2': 21,
                          '-3': 22, ',': 23, '.': 24, 'first': 25, 'second': 26, 'third': 27, 'fourth': 28, 'quadrant': 29} #
        elif dataset == 'cater-gen-v2':
            self.vocab = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, 'the': 3, 'cone': 4, 'snitch': 5, 'is': 6,
                          'sliding': 7, 'picked': 8, 'placed': 9, 'containing': 10, 'and': 11, 'to': 12,
                          'up': 13, 'sphere': 14, 'cylinder': 15, 'cube': 16, 'small': 17, 'medium': 18,
                          'large': 19, 'metal': 20, 'rubber': 21, 'gold': 22, 'gray': 23, 'red': 24,
                          'blue': 25, 'green': 26, 'brown': 27, 'purple': 28, 'cyan': 29, 'yellow': 30,
                          '(': 31, ')': 32, '1': 33, '2': 34, '3': 35, '-1': 36, '-2': 37,
                          '-3': 38, ',': 39, '.': 40, 'rotating': 41, 'while': 42, 'contained': 43, 'still': 44,
                          'first': 45, 'second': 46, 'third': 47, 'fourth': 48, 'quadrant': 49}
        self.padding_idx = self.vocab['[PAD]']

    def __len__(self):
        return len(self.anno)

    def sent2matrix(self, x):
        words = nltk.word_tokenize(x)
        m = np.int32(np.zeros((1, len(words))))

        for i in range(len(words)):
            m[0, i] = self.vocab[words[i]]
        return m

    def tokens2sent(self, tokens):
        vocab_reverse = {}
        for word in self.vocab:
            index = self.vocab[word]
            vocab_reverse[index] = word
        text = ""
        for i in range(tokens.shape[0]):
            text = text + " " + vocab_reverse[tokens[i]]
        return text

    def __getitem__(self, idx):
        video_path = self.anno[str(idx)]['video']
        video_path = os.path.join(self.data_root, video_path)
        caption = self.anno[str(idx)]['caption']
        caption_tokens = self.sent2matrix(caption)
        caption_tokens = np.insert(caption_tokens, 0, self.vocab['[CLS]'])
        caption_tokens = np.append(caption_tokens, self.vocab['[SEP]'])
        caption_tokens = torch.tensor(caption_tokens, dtype=torch.long)

        vid = VideoReader(video_path)
        frame_num = len(vid)

        speed = random.random()
        sample_interval = max(3.0, speed * (self.sample_speed[-1] - self.sample_speed[0]) + self.sample_speed[0])
        choice_idx = np.floor(np.linspace(0, frame_num-1, round(frame_num / sample_interval), endpoint=True)).astype(np.int32)
        images = vid.get_batch(list(choice_idx)).asnumpy()
        images = images[:self.frames_length]
        images = self.transform([Image.fromarray(i) for i in images]).permute(1, 0, 2, 3)
        if images.shape[0] < self.frames_length:
            images = torch.cat([images, images[-1].unsqueeze(0).repeat(self.frames_length - images.shape[0], 1, 1, 1)], dim=0)

        if self.randomness:
            choice_idx = np.floor(np.linspace(0, frame_num - 1, self.frames_length, endpoint=True)).astype(np.int32)
            images_ori = vid.get_batch(list(choice_idx)).asnumpy()
            images_ori = self.transform([Image.fromarray(i) for i in images_ori]).permute(1, 0, 2, 3)
            images = torch.cat([images, images_ori], 0)

        return images, caption_tokens, torch.tensor(speed, dtype=torch.float), torch.tensor(len(caption_tokens), dtype=torch.long)

    def collate_fn(self, data):

        caption_tokens = torch.nn.utils.rnn.pad_sequence(
            [d[1] for d in data],
            batch_first=True,
            padding_value=self.padding_idx,
        )

        return torch.stack([d[0] for d in data], dim=0), caption_tokens, torch.stack([d[2] for d in data], dim=0), torch.stack([d[3] for d in data], dim=0)

class CATER4VQVAE(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        image_transform=None,
    ):
        self.reader = LmdbReader2(os.path.join(data_root, f'vqvae_{split}.lmdb'))  # To improve the efficiency, we gather all videos and split into images, then store as vqvae_train.lmdb and vqvae_test.lmdb
        self.transform = image_transform

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        image = self.reader[idx]
        image = self.transform(Image.fromarray(image))
        return image

class LmdbReader2(Dataset):
    def __init__(self, lmdb_path: str, shuffle: bool = True, percentage: float = 100):
        self.lmdb_path = lmdb_path
        self.shuffle = shuffle

        assert percentage > 0, "Cannot load dataset with 0 percent original size."
        self.percentage = percentage

        env = lmdb.open(
            self.lmdb_path, subdir=False, readonly=True, lock=False,
            readahead=False, map_size=1099511627776 * 2,
        )
        self.db_txn = env.begin()

        self._keys = [
            f"{i}".encode("ascii") for i in range(env.stat()["entries"])
        ]
        if percentage < 100.0:
            retain_k: int = int(len(self._keys) * percentage / 100.0)
            random.shuffle(self._keys)
            self._keys = self._keys[:retain_k]

        self.shuffle_seed = 0

    def set_shuffle_seed(self, seed: int):
        r"""Set random seed for shuffling data."""
        self.shuffle_seed = seed

    def get_keys(self) -> List[bytes]:
        r"""Return list of keys, useful while saving checkpoint."""
        return self._keys

    def set_keys(self, keys: List[bytes]):
        r"""Set list of keys, useful while loading from checkpoint."""
        self._keys = keys

    def __getstate__(self):
        r"""
        This magic method allows an object of this class to be pickable, useful
        for dataloading with multiple CPU workers. :attr:`db_txn` is not
        pickable, so we remove it from state, and re-instantiate it in
        :meth:`__setstate__`.
        """
        state = self.__dict__
        state["db_txn"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state

        env = lmdb.open(
            self.lmdb_path, subdir=False, readonly=True, lock=False,
            readahead=False, map_size=1099511627776 * 2,
        )
        self.db_txn = env.begin()

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, idx: int):
        datapoint_pickled = self.db_txn.get(self._keys[idx])
        image = pickle.loads(datapoint_pickled)

        return image

class MNIST4VQVAE(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        image_transform=None,
    ):
        self.reader = LmdbReader(data_root + split + '.lmdb')
        self.transform = image_transform

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        images_raw, caption = self.reader[idx]
        frame_num = images_raw.shape[0]
        choice_idx = random.choice(range(frame_num))
        images_raw = images_raw[choice_idx]
        if self.transform is not None:
            image = self.transform(Image.fromarray(images_raw[0]))
        else:
            image = images_raw / 255. - 0.5
            image = torch.tensor(image, dtype=torch.float)

        return image

