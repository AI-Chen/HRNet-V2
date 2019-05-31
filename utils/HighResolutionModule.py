from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.training import moving_averages

import tensorflow as tf
import  logging
from utils import Models_Block as block_Model
from utils import Models_Config as config

logger = logging.getLogger(__name__)

class HighResolutionModule():
    def __init__(self,stage,_is_training,num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        self.stage = stage
        self._is_training = _is_training
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output

    def forword(self,layer,num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        self._check_branches(num_branches,num_blocks, num_inchannels, num_channels)
        Result = self._make_branches(layer,num_branches, blocks, num_blocks, num_channels)
        Result = self._make_fuse_layers(Result)
        channel = []
        for i in range(len(Result)):
            channel.append(Result[i].get_shape().as_list()[3])
        return Result,channel



    def _check_branches(self, num_branches,num_blocks,
                            num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_branches(self, layer,num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(
                self._make_one_branch(layer[i], i, block, num_blocks, num_channels))
        return branches

    def _make_one_branch(self,x,branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * 1:
            downsample = block_Model.conv
        x = block(x, is_training=self._is_training, block_name=self.stage + '_Botteleneck0' + str(branch_index), outplanes=num_channels[branch_index] * 1,
                  stride=stride, downsample=downsample)
        for i in range(1, num_blocks[branch_index]):
            x = block(x, is_training=self._is_training, block_name=self.stage + 'Botteleneck' + str(i) + str(branch_index), outplanes=num_channels[branch_index])
        return x

    def _make_fuse_layers(self,x):
        if self.num_branches == 1:
            return x
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            temp_layer = []
            for j in range(num_branches):
                if j > i:
                    with tf.variable_scope(self.stage + '_fuse_layers' + str(i)+str(j)):
                        temp = block_Model.conv(x[j], 1, num_inchannels[i], 'conv_', stride=1)
                        temp = block_Model.batch_norm(temp, self._is_training)
                        temp = tf.image.resize_bilinear(temp, x[i].get_shape().as_list()[1:3])
                        temp_layer.append(temp)
                elif j == i:
                    temp_layer.append(x[j])
                else:
                    with tf.variable_scope(self.stage + '_fuse_layers' + str(i) + str(j)):
                        temp = block_Model.conv(x[j], 3, num_inchannels[i], 'conv_', stride=2)
                        temp = block_Model.batch_norm(temp, self._is_training)
                        temp_layer.append(temp)
            for k in range(len(temp_layer)):
                fuse = 0
                fuse = fuse + temp_layer[k]
            fuse_layers.append(fuse)
        return fuse_layers
