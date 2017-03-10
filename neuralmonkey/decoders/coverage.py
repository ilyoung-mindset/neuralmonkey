from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.trainers.generic_trainer import Objective
from neuralmonkey.decoders.decoder import Decoder

import tensorflow as tf


def coverage_objective(
        name: str,
        encoder: ModelPart,
        decoder: Decoder,
        weight: float,
        use_fertility: bool=False) -> Objective:

    attn_object = decoder.get_attention_object(encoder, train_mode=True)

    # batch x decoder time
    output_padding = tf.transpose(decoder.train_padding, perm=[1, 0])

    # batch x decoder time x encoder time
    alignments = tf.transpose(
        tf.pack(attn_object.attentions_in_time), perm=[1, 0, 2])

    # mask out what is not in decoder and sum over the decoder time
    #  => we get encoder coverage
    coverage = tf.reduce_sum(alignments * output_padding, 1)

    if use_fertility:
        raise NotImplementedError("Fertility is not implemented yet.")
    else:
        loss = tf.square(1 - coverage)

    return Objective(name, decoder, loss, None, weight)
