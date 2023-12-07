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


class ErBuf(ContinualModel):
    NAME = 'er_buf'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    

    def __init__(self, backbone, loss, args, transform):
        super(ErBuf, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

        self.label_added = []
        self.SAMPLE_PER_CLASS = 200

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]
        input_labels = labels

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
        
        unique_label, count = torch.unique(input_labels,return_counts=True)
        images = None
        saved_labels = None
        for label, label_count in zip(unique_label,count):
            

            image = np.load(f'datasets/cifar_10_generated/class_{label}.npz')
            indices = torch.randperm(len(image['samples']))[:label_count]

            if (images is not None) and (saved_labels is not None):
                norm_images = torch.tensor(image['samples'][indices])/255
                images = torch.cat((images,norm_images), dim=0)
                processed_labels = torch.argmax(torch.tensor(image['label'][indices]),dim=1)
                saved_labels = torch.cat((saved_labels,processed_labels),  dim=0)
            else:
                images = torch.tensor(image['samples'][indices])/255
                saved_labels = torch.argmax(torch.tensor(image['label'][indices]),dim=1)


        if images is not None:
            self.buffer.add_data(examples=images.permute(0,3,1,2),
                                labels=saved_labels)

        return loss.item()

