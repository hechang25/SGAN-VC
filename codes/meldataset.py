
import os
import time
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(1)
random.seed(1)


def make_dataset(dir):
    audios = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for pid in os.listdir(os.path.join(dir, "mel")):
        for file_name in os.listdir(os.path.join(dir, "mel", pid)):
            mel_path = os.path.join(dir, "mel", pid, file_name)
            mel = np.load(mel_path)
            file_name_dict = dict(file_name=file_name, mel=mel)
            audios.append((file_name_dict, pid))

    return audios


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, mean, std, validation=False):
        self.audios = make_dataset(data_root)
        self.targets = np.asarray([s[1] for s in self.audios])
        self.img_num = len(self.audios)

        self.mean, self.std = mean, std
        self.mel_min, self.mel_max = -11.512925, 2.2482383
        self.validation = validation
        self.max_mel_length = 224
        

    def __len__(self):
        return len(self.audios)

    def _get_pos_sample(self, target, audio_idx):
        pos_index = np.argwhere(self.targets == target)
        pos_index = pos_index.flatten()
        pos_index = np.setdiff1d(pos_index, audio_idx)
        rand = random.randint(0, len(pos_index) - 1)
        return self.audios[pos_index[rand]][0], self.audios[pos_index[rand]][1]

    def __getitem__(self, idx):
        sample_dict, label = self.audios[idx]
        mel_tensor = self._process_data(sample_dict["mel"])

        audio_idx, (ref_dict, ref_label) = random.choice(list(enumerate(self.audios)))
        ref_mel_tensor = self._process_data(ref_dict["mel"])
        ref2_data, ref_label2 = self._get_pos_sample(ref_label, audio_idx)
        assert ref_label == ref_label2

        ref2_mel_tensor = self._process_data(ref2_data["mel"])
        label = torch.tensor(int(label))
        ref_label = torch.tensor(int(ref_label))

        return mel_tensor, label, ref_mel_tensor, ref2_mel_tensor, ref_label

    def _process_data(self, mel_tensor):
        mel_tensor = torch.from_numpy(mel_tensor).float()    
        mel_tensor = (mel_tensor - self.mean) / self.std
        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]
        return mel_tensor


class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave
        self.max_mel_length = 224
        
    def __call__(self, batch):
        batch_size = len(batch)
        nmels = batch[0][0].size(0)
        mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        labels = torch.zeros((batch_size)).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref2_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()

        for bid, (mel, label, ref_mel, ref2_mel, ref_label) in enumerate(batch):
            mel_size = mel.size(1)
            mels[bid, :, :mel_size] = mel

            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel

            ref2_mel_size = ref2_mel.size(1)
            ref2_mels[bid, :, :ref2_mel_size] = ref2_mel

            labels[bid] = label
            ref_labels[bid] = ref_label

        mels, ref_mels, ref2_mels = mels.unsqueeze(1), ref_mels.unsqueeze(1), ref2_mels.unsqueeze(1)
        return mels, labels, ref_mels, ref2_mels, ref_labels


def build_dataloader(path_list,
                     mean,
                     std,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={}):
    dataset = MelDataset(path_list, mean=mean, std=std, validation=validation)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader
