from typing import Optional

from neuralmonkey.encoders.attentive import Attentive
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.trainers.generic_trainer import Objective
from neuralmonkey.decoders.decoder import Decoder
from neuralmonkey.dataset import Dataset

import tensorflow as tf


class FertilityModel(ModelPart):
    def __init__(self,
                 name: str,
                 encoder: Attentive,
                 save_checkpoint: Optional[str]=None,
                 load_checkpoint: Optional[str]=None) -> None:
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        hidden_states = encoder.attention_tensor
        state_size = hidden_states.get_shape()[2].value

        with tf.variable_scope(name):
            weights = tf.get_variable("fertility_weight",
                                      shape=[1, state_size, 1])
            bias = tf.get_variable("fertility_bias", shape=[])

            self.fertilities = tf.squeeze(tf.nn.conv1d(
                hidden_states, weights, stride=1, padding='SAME') + bias, [2])

    def feed_dict(self, dataset: Dataset, train: bool=False) -> FeedDict:
        return {}


def compute_coverage(name: str, encoder: Attentive,
                     decoder: Decoder) -> tf.Tensor:
    with tf.variable_scope(name):
        attn_object = decoder.get_attention_object(encoder, train_mode=True)

        # batch x decoder time
        output_padding = tf.expand_dims(
            tf.transpose(decoder.train_padding, perm=[1, 0]), -1)

        # batch x decoder time x encoder time
        alignments = tf.transpose(
            tf.pack(attn_object.attentions_in_time), perm=[1, 0, 2])

        # mask out what is not in decoder and sum over the decoder time
        #  => we get encoder coverage
        coverage = tf.reduce_sum(alignments * output_padding, 1)
        return coverage


def coverage_objective(
        name: str,
        encoder: Attentive,
        decoder: Decoder,
        weight: float,
        fertility_model: Optional[FertilityModel]) -> Objective:

    coverage = compute_coverage(name, encoder, decoder)

    if fertility_model is not None:
        loss = tf.square(fertility_model.fertilities - coverage)
    else:
        loss = tf.square(1 - coverage)

    return Objective(name, decoder, loss, None, weight)
