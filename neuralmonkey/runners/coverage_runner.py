from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf

from neuralmonkey.encoders.attentive import Attentive
from neuralmonkey.decoders.decoder import Decoder
from neuralmonkey.decoders.coverage import FertilityModel, compute_coverage
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.runners.base_runner import (BaseRunner, Executable,
                                              ExecutionResult, NextExecute)


class CoverageExecutable(Executable):
    def __init__(self,
                 all_coders: List[ModelPart],
                 mask: tf.Tensor,
                 coverage: tf.Tensor,
                 fertility: Optional[tf.Tensor]) -> None:
        self._all_coders = all_coders

        self._fetches = {"coverage": coverage,
                         "mask": mask}
        if fertility is not None:
            self._fetches["fertility"] = fertility

        self.result = None  # type: Optional[ExecutionResult]

    def next_to_execute(self) -> NextExecute:
        return self._all_coders, self._fetches, {}

    def collect_results(self, results: List[Dict]) -> None:
        coverage = results[0]["coverage"]
        masks = results[0]["mask"]
        if "fertility" in results[0]:
            fertility = results[0]["fertility"]
        else:
            fertility = np.ones(coverage.shape)

        for res in results[1:]:
            coverage += res["coverage"]
            if "fertility" in res:
                fertility += res["fertility"]

        coverage /= len(results)
        fertility /= len(results)

        coverage_loss = np.mean(np.square(coverage - fertility))

        output = []
        for sent_coverage, sent_fertility, mask in zip(
                coverage.tolist(), fertility.tolist(), masks.tolist()):
            sent_list = []
            for coverage, fertility, weight in zip(
                    sent_coverage, sent_fertility, mask):
                if weight != 0:
                    sent_list.append("{:.2f}/{:.2f}".format(
                        coverage, fertility))

            output.append(sent_list)

        self.result = ExecutionResult(
            outputs=output,
            losses=[coverage_loss],
            scalar_summaries=None, histogram_summaries=None,
            image_summaries=None)


class CoverageRunner(BaseRunner):
    def __init__(self,
                 output_series: str,
                 decoder: Decoder,
                 encoder: Attentive,
                 fertility: Optional[FertilityModel]) -> None:
        super(CoverageRunner, self).__init__(output_series, decoder)

        self._coverage = compute_coverage(
            "coverage_{}".format(output_series), encoder, decoder)
        self._attention_mask = encoder.attention_mask
        if fertility is not None:
            self._fertility = fertility.fertilities
        else:
            self._fertility = None

    def get_executable(self, compute_losses=False,
                       summaries=True) -> CoverageExecutable:
        return CoverageExecutable(self.all_coders, self._attention_mask,
                                  self._coverage, self._fertility)

    @property
    def loss_names(self) -> List[str]:
        return ["avg_coverage"]
