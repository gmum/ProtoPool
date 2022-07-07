from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax


from resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features
import numpy as np

from utils import compute_proto_layer_rf_info_v2

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}


class PrototypeChooser(nn.Module):

    def __init__(self, num_prototypes: int, num_descriptive: int, num_classes: int,
                 use_thresh: bool = False, arch: str = 'resnet34', pretrained: bool = True,
                 add_on_layers_type: str = 'linear', prototype_activation_function: str = 'log',
                 proto_depth: int = 128, use_last_layer: bool = False, inat: bool = False) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = 1e-4
        self.num_descriptive = num_descriptive
        self.num_prototypes = num_prototypes
        self.proto_depth = proto_depth
        self.prototype_shape = (self.num_prototypes, self.proto_depth, 1, 1)
        self.use_thresh = use_thresh
        self.arch = arch
        self.pretrained = pretrained
        self.prototype_activation_function = prototype_activation_function
        self.inat = inat
        if self.use_thresh:
            self.alfa = Parameter(torch.Tensor(1, num_classes, num_descriptive))
            nn.init.xavier_normal_(self.alfa, gain=1.0)
        else:
            self.alfa = 1
            self.beta = 0

        self.proto_presence = torch.zeros(num_classes, num_prototypes, num_descriptive)  # [c, p, n]
        # for j in range(num_classes):
        #     for k in range(num_descriptive):
        #         self.proto_presence[j, j * num_descriptive + k, k] = 1
        self.proto_presence = Parameter(self.proto_presence, requires_grad=True)
        nn.init.xavier_normal_(self.proto_presence, gain=1.0)
        if self.inat:
            self.features = base_architecture_to_features['resnet50'](pretrained=pretrained, inat=True)
        else:
            self.features = base_architecture_to_features[self.arch](pretrained=pretrained)

        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in self.features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in self.features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        if add_on_layers_type == 'bottleneck':
            raise NotImplementedError
        else:
            add_on_layers = [
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1],
                          kernel_size=1),
                # nn.ReLU(),
                # nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid(),
            ]

            self.add_on_layers = nn.Sequential(*add_on_layers)

        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)

        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        # initial weights
        for m in self.add_on_layers.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.use_last_layer = use_last_layer
        if self.use_last_layer:
            self.prototype_class_identity = torch.zeros(self.num_descriptive * self.num_classes, self.num_classes)

            for j in range(self.num_descriptive * self.num_classes):
                self.prototype_class_identity[j, j // self.num_descriptive] = 1
            self.last_layer = nn.Linear(self.num_descriptive * self.num_classes, self.num_classes, bias=False)
            positive_one_weights_locations = torch.t(self.prototype_class_identity)
            negative_one_weights_locations = 1 - positive_one_weights_locations

            correct_class_connection = 1
            incorrect_class_connection = 0 # -0.5
            self.last_layer.weight.data.copy_(
                correct_class_connection * positive_one_weights_locations
                + incorrect_class_connection * negative_one_weights_locations)
        else:
            self.last_layer = nn.Identity()

    def fine_tune_last_only(self):
        for p in self.features.parameters():
            p.requires_grad = False
        for p in self.add_on_layers.parameters():
            p.requires_grad = False
        self.prototype_vectors.requires_grad = False
        self.proto_presence.requires_grad = False
        for p in self.last_layer.parameters():
            p.requires_grad = True

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.features(x)
        x = self.add_on_layers(x)
        return x

    def forward(self, x: torch.Tensor, gumbel_scale: int = 0) -> \
            Tuple[torch.Tensor, torch.LongTensor]:
        if gumbel_scale == 0:
            proto_presence = torch.softmax(self.proto_presence, dim=1)
        else:
            proto_presence = gumbel_softmax(self.proto_presence * gumbel_scale, dim=1, tau=0.5)

        distances = self.prototype_distances(x)  # [b, C, H, W] -> [b, p, h, w]

        '''
        we cannot refactor the lines below for similarity scores
        because we need to return min_distances
        '''
        # global min pooling
        min_distances = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3])).squeeze()  # [b, p]
        avg_dist = F.avg_pool2d(distances, kernel_size=(distances.size()[2],
                                                        distances.size()[3])).squeeze()  # [b, p]
        min_mixed_distances = self._mix_l2_convolution(min_distances, proto_presence)  # [b, c, n]
        avg_mixed_distances = self._mix_l2_convolution(avg_dist, proto_presence)  # [b, c, n]
        x = self.distance_2_similarity(min_mixed_distances)  # [b, c, n]
        x_avg = self.distance_2_similarity(avg_mixed_distances)  # [b, c, n]
        x = x - x_avg
        # x = self.distance_2_similarity(min_distances)
        if self.use_last_layer:
            x = self.last_layer(x.flatten(start_dim=1))
        else:
            x = x.sum(dim=-1)
        return x, min_distances, proto_presence  # [b,c,n] [b, p] [c, p, n]

    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=self.prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances

    def _mix_l2_convolution(self, distances, proto_presence):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        # distances [b, p]
        # proto_presence [c, p, n]
        mixed_distances = torch.einsum('bp,cpn->bcn', distances, proto_presence)

        return mixed_distances  # [b, c, n]

    def prototype_distances(self, x):
        '''
        x is the raw input
        '''
        conv_features = self.conv_features(x)
        distances = self._l2_convolution(conv_features)  # [b, p, h, w]
        return distances  # [b, n, h, w], [b, p, h, w]

    def distance_2_similarity(self, distances):  # [b,c,n]
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            if self.use_thresh:
                distances = distances  # * torch.exp(self.alfa)  # [b, c, n]
            return 1 / (distances + 1)
        else:
            raise NotImplementedError

    def get_map_class_to_prototypes(self):
        pp = gumbel_softmax(self.proto_presence * 10e3, dim=1, tau=0.5).detach()
        return np.argmax(pp.cpu().numpy(), axis=1)

    def __repr__(self):
        res = super(PrototypeChooser, self).__repr__()
        return res
