# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class Er(ContinualModel):
    NAME = 'er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    

    def __init__(self, backbone, loss, args, transform):
        super(Er, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

        self.label_added = []
        self.SAMPLE_PER_CLASS = 200

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()
        
        # unique_label = torch.unique(labels)
        # images = None
        # saved_labels = None
        # for label in unique_label:

        #     if label not in self.label_added:
        #         self.label_added.append(int(label))
        #         image = np.load(f'datasets/cifar_100_generated/class_{label}.npz')

        #         if (images is not None) and (saved_labels is not None):
        #             images = torch.cat((images,torch.tensor(image['samples'][:self.SAMPLE_PER_CLASS], dtype=torch.float32)), dim=0)
        #             saved_labels = torch.cat((saved_labels,torch.tensor(image['label'][:self.SAMPLE_PER_CLASS], dtype=torch.float32)), dim=0)
        #         else:
        #             images = torch.tensor(image['samples'][:self.SAMPLE_PER_CLASS], dtype=torch.float32)
        #             saved_labels = torch.tensor(image['label'][:self.SAMPLE_PER_CLASS], dtype=torch.float32)



        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()

