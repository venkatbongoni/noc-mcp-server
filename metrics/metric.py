# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025 Cisco Systems.
# Author: Timo Koehler
#
# Licensed under the MIT License. See LICENSE file in the project root.
#
# Portions of this software are protected by U.S. Patent No. [Patent Number], 
# held by Cisco Systems.
# -----------------------------------------------------------------------------

import dspy
from rouge_score import rouge_scorer


class Accuracy(dspy.Signature):
    """Evaluate how well the predicted answer correctly identifies and elaborates
    on all key aspects of the gold answer."""
    answer: str = dspy.InputField(desc="the gold answer")
    prediction: str = dspy.InputField(desc="the predicted answer")
    accuracy: float = dspy.OutputField(ge=1.0, le=10.0, desc="accuracy value from 1.0 (prediction does not match the answer) to 10.0 (total match).")


class LLMMetric(dspy.Module):
    """Please act as an impartial judge and evaluate the quality of the predicted
    answers provided by multiple AI assistants to the user question displayed
    below. You should choose the assistant that offers a better user experience by
    interacting with the user more effectively and efficiently, and providing a
    correct final response to the user's question.
    Rules:
    1. Avoid Position Biases: Ensure that the order in which the responses were
    presented does not influence your decision. Evaluate each response on its own
    merits.
    2. Length of Responses: Do not let the length of the responses affect your
    evaluation. Focus on the quality and relevance of the response. A good response
    is targeted and addresses the user's needs effectively, rather than simply being
    detailed.
    3. Objectivity: Be as objective as possible. Consider the user's perspective and
    overall experience with each assistant.
    4. Prioritize the accuracy and clarity of conveyed information: Phrases conveying
    the same essential meaning should be evaluated with the same score."""
    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.accuracy = dspy.ChainOfThought(Accuracy, max_retries=5)

    def forward(self, gold, pred, trace=None):
        with dspy.context(lm=self.lm):
            accuracy = self.accuracy(answer=gold.answer, prediction=pred.answer)

        score = round(accuracy.accuracy * 0.1, 3)
        if trace is not None:
            return score > 0.4
        return score


class RougeMetric:
    def __init__(self, use_stemmer=True):
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=use_stemmer)

    def metric(self, example: str, prediction: str, trace=None) -> float:
        score = self.scorer.score(example, prediction)
        precision = score['rougeL'].precision
        precision = round(precision, 3)
        if trace is not None:
            return precision > 0.4
        return precision

    def __call__(self, example: str, prediction: str, trace=None) -> float:
        return self.metric(example, prediction, trace)


class SemanticF1Metric:
    def __init__(self, decompositional=True):
        self.metric = dspy.evaluate.SemanticF1(decompositional=decompositional)

    def __call__(self, example, pred, trace=None):
        return self.metric(example.copy(response=example.answer), pred.copy(response=pred.answer), trace)
