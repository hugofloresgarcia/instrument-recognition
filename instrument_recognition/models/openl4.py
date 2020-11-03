import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from instrument_recognition.models.timefreq import Melspectrogram
from instrument_recognition.utils.train_utils import Hook, load_weights

class LayerUnfreezeCallback(pl.callbacks.base.Callback):
    def __init__(self, unfreeze_epoch=10):
        self.unfreeze_epoch = unfreeze_epoch


    def on_epoch_end(self, trainer, module):
        epoch = trainer.current_epoch

        if epoch == self.unfreeze_epoch:
            module.openl3.is_frozen = False
            module.openl3.freeze_up_to('maxpool')
            module.openl3.unfreeze_starting_at('conv2d_6')

class OpenL3Mel128(nn.Module):

    def __init__(self, weight_file='./instrument_recogntion/weights/openl3/openl3_music_6144_no_mel_layer_pytorch_weights', 
                maxpool_kernel=(16, 24), 
                maxpool_stride=(4, 8)):
        super(OpenL3Mel128, self).__init__()

        self._weights_dict = load_weights(weight_file)

        self.is_frozen = False

        self.batch_normalization_1 = self.__batch_normalization(2, 'batch_normalization_1', num_features=1, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_1 = self.__conv(2, name='conv2d_1', in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_2 = self.__batch_normalization(2, 'batch_normalization_2', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_2 = self.__conv(2, name='conv2d_2', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_3 = self.__batch_normalization(2, 'batch_normalization_3', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_3 = self.__conv(2, name='conv2d_3', in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_4 = self.__batch_normalization(2, 'batch_normalization_4', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_4 = self.__conv(2, name='conv2d_4', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_5 = self.__batch_normalization(2, 'batch_normalization_5', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_5 = self.__conv(2, name='conv2d_5', in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_6 = self.__batch_normalization(2, 'batch_normalization_6', num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_6 = self.__conv(2, name='conv2d_6', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_7 = self.__batch_normalization(2, 'batch_normalization_7', num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_7 = self.__conv(2, name='conv2d_7', in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_8 = self.__batch_normalization(2, 'batch_normalization_8', num_features=512, eps=0.0010000000474974513, momentum=0.0)
        self.audio_embedding_layer = self.__conv(2, name='audio_embedding_layer', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel, stride=maxpool_stride, padding=0, ceil_mode=False)

    def unfreeze_layer(self, layer_name):
        """
        freeze model from input up to (and including) a layer specified by layer_name
        a warning will be raised if you use an incorrect layer name
        """
        for name, module in self.named_modules():
            if layer_name in name:
                print(f'setting {name} to eval')
                module.eval()
                break

        for name, param in self.named_parameters():
            if layer_name in name:
                param.requires_grad = False
                print(f'freezing {name}')
                break

    def freeze_up_to(self, layer_name):
        """
        freeze model from input up to (and including) a layer specified by layer_name
        a warning will be raised if you use an incorrect layer name
        """
        for name, module in self.named_modules():
            module.eval()
            print(f'setting {name} to eval')
            if layer_name in name:
                break

        for name, param in self.named_parameters():
            param.requires_grad = False
            print(f'freezing {name}')
            if layer_name in name:
                break
            
    def unfreeze_starting_at(self, layer_name):
        """
        unfreeze starting at a  layer specified w layer_name
        """
        unfreeze=False
        self.train()
        #TODO: it could be that the batchnorms that are around frozen areas need to be frozen with .eval() as well
        for name, module in self.named_modules():
            if layer_name in name:
                unfreeze=True

            if 'batch' in name and not unfreeze:
                print(f'setting {name} to eval')
                module.eval()
            if unfreeze:
                print(f'setting {name} to train')
                module.train()

        for name, param in self.named_parameters():
            if layer_name in name:
                unfreeze=True

            if unfreeze:
                print(f'unfreezing: {name}')
                param.requires_grad=True

    def forward(self, x):
        batch_normalization_1 = self.batch_normalization_1(x)
        conv2d_1_pad    = F.pad(batch_normalization_1, (1, 1, 1, 1))
        conv2d_1        = self.conv2d_1(conv2d_1_pad)
        batch_normalization_2 = self.batch_normalization_2(conv2d_1)
        activation_1    = F.relu(batch_normalization_2)
        conv2d_2_pad    = F.pad(activation_1, (1, 1, 1, 1))
        conv2d_2        = self.conv2d_2(conv2d_2_pad)
        batch_normalization_3 = self.batch_normalization_3(conv2d_2)
        activation_2    = F.relu(batch_normalization_3)
        max_pooling2d_1, max_pooling2d_1_idx = F.max_pool2d(activation_2, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        conv2d_3_pad    = F.pad(max_pooling2d_1, (1, 1, 1, 1))
        conv2d_3        = self.conv2d_3(conv2d_3_pad)
        batch_normalization_4 = self.batch_normalization_4(conv2d_3)
        activation_3    = F.relu(batch_normalization_4)
        conv2d_4_pad    = F.pad(activation_3, (1, 1, 1, 1))
        conv2d_4        = self.conv2d_4(conv2d_4_pad)
        batch_normalization_5 = self.batch_normalization_5(conv2d_4)
        activation_4    = F.relu(batch_normalization_5)
        max_pooling2d_2, max_pooling2d_2_idx = F.max_pool2d(activation_4, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        conv2d_5_pad    = F.pad(max_pooling2d_2, (1, 1, 1, 1))
        conv2d_5        = self.conv2d_5(conv2d_5_pad)
        batch_normalization_6 = self.batch_normalization_6(conv2d_5)
        activation_5    = F.relu(batch_normalization_6)
        conv2d_6_pad    = F.pad(activation_5, (1, 1, 1, 1))
        conv2d_6        = self.conv2d_6(conv2d_6_pad)
        batch_normalization_7 = self.batch_normalization_7(conv2d_6)
        activation_6    = F.relu(batch_normalization_7)
        max_pooling2d_3, max_pooling2d_3_idx = F.max_pool2d(activation_6, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        conv2d_7_pad    = F.pad(max_pooling2d_3, (1, 1, 1, 1))
        conv2d_7        = self.conv2d_7(conv2d_7_pad)
        batch_normalization_8 = self.batch_normalization_8(conv2d_7)
        activation_7    = F.relu(batch_normalization_8)
        audio_embedding_layer_pad = F.pad(activation_7, (1, 1, 1, 1))
        audio_embedding_layer = self.audio_embedding_layer(audio_embedding_layer_pad)
        max_pooling2d_4 =  self.maxpool(audio_embedding_layer)
        # flatten_1       = max_pooling2d_4.view(max_pooling2d_4.size(0), -1)
            
        return max_pooling2d_4

    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 0 or dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        if 'scale' in _weights_dict[name]:
            layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['scale']))
        else:
            layer.weight.data.fill_(1)

        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        else:
            layer.bias.data.fill_(0)

        layer.state_dict()['running_mean'].copy_(torch.from_numpy(_weights_dict[name]['mean']))
        layer.state_dict()['running_var'].copy_(torch.from_numpy(_weights_dict[name]['var']))
        return layer

class OpenL4Student(nn.Module):
    
    def __init__(self, sr=48000, dropout=0.1):
        super().__init__()
        self.sr = sr

        self.filters = Melspectrogram(sr=self.sr)

        self.openl3 = OpenL3Mel128()
  
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(), 
            nn.Dropout(p=dropout),

            nn.BatchNorm1d(128),
            nn.Linear(128, 19), 
        )

    def forward(self, x, return_all=True):
        # x should be raw audio shape (batch, channels, time)

        # pass through filterbank, get time freq representation
        tf_repr = self.filters(x) # shape (batch, channels, frequency, time)

        # pass through openl3 embedding, get embedding
        embedding = self.openl3(tf_repr) # shape (batch, embedding)

        # pass through classifier and get class probabilities
        probs = self.classifier(self.flatten(embedding))

        if return_all:
            return tf_repr, embedding, probs    
        else:
            return probs
 
class MonoSEDTrainingModule(pl.LightningModule):

    def __init__(self, hparams):

        self.datamodule = MonoSEDDataModule(
            batch_size=hparams.batch_size, 
            num_worker=hparams.num_dataloader_workers
        )

        self.classes = self.datamodule.classes

        self.model = OpenL4Student(
            sr=self.hparams.sr
        )

        if hparams.teacher:
            self.teacher = torchopenl3.OpenL3Mel128()
    
    @staticmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument('--sr',     efault=16000,           type=int)
        parser.add_argument('--dropout',        default=0.1,    type=int)
        parser.add_argument('--mixup',          default=False,   type=str2bool)
        parser.add_argument('--openl3_unfreeze_epoch', default=6, type=int)
        parser.add_argument('--teacher', default=False, type=str2bool)
        return parser

    def training_step(self, batch, batch_idx):
        # should be receiving batch of 1s of audio      
        raise NotImplementedError  