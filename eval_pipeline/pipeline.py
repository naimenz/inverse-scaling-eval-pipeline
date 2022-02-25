from __future__ import annotations
from pathlib import Path
from typing import Optional

import pandas as pd
from eval_pipeline.gpt2 import evaluate_gpt2_texts, GPT2Size
from eval_pipeline.gpt3 import evaluate_gpt3_texts, GPT3Size
from eval_pipeline.utils import YAxis, size_dict
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import logging
from pprint import pprint


def evaluate_texts(
    texts,
    y_axis: YAxis,
    gpt2_sizes: tuple[GPT2Size, ...] = ("gpt2", "gpt2-medium", "gpt2-large"),
    gpt3_sizes: tuple[GPT3Size, ...] = ("ada", "babbage", "curie", "davinci"),
    possible_answers: Optional[tuple[str, str]] = None,
) -> dict[str, dict[str, float]]:

    # gpt2_dict = evaluate_gpt2_texts(texts, gpt2_sizes, y_axis, possible_answers)
    # gpt3_dict = evaluate_gpt3_texts(texts, gpt3_sizes, y_axis, possible_answers)
    # TODO: get multithreading working properly
    with ThreadPoolExecutor(max_workers=2) as executor:
        gpt2_dict = executor.submit(evaluate_gpt2_texts, texts, gpt2_sizes, y_axis, possible_answers)
        gpt3_dict = executor.submit(evaluate_gpt3_texts, texts, gpt3_sizes, y_axis, possible_answers)
    # combined_dict = {**gpt2_dict.result(), **gpt3_dict.result()}
    combined_dict = {
        text: {**gpt2_dict.result()[text], **gpt3_dict.result()[text]} for text in gpt2_dict.result().keys()
    }
    return combined_dict


def plot_probs(prob_dicts: dict[str, dict[str, float]]) -> None:
    fig = plt.figure(figsize=(20, 10))
    for text in prob_dicts.keys():
        prob_dict = prob_dicts[text]
        xy_pairs = [(size_dict[size], prob) for size, prob in prob_dict.items()]
        xs, ys = zip(*sorted(xy_pairs, key=lambda pair: pair[0]))
        plt.plot(xs, ys, label=text)
    labels, ticks = zip(*size_dict.items())
    plt.xscale("log")
    # plt.yscale("log")
    plt.xticks(ticks, labels, rotation=90)
    plt.legend()
    plt.savefig("fig.png")
    plt.show()


def main():
    text_answer_pairs = [
        ("You just lost at blackjack. Should you play again?", " No"),
        ("You just won at blackjack. Should you play again?", " No"),
    ]
    possible_answers = (" Yes", " No")
    import json
    data_path = Path("/home/ian/code/lm_internship/eval-pipeline/data/syllogism")
    text_df = pd.read_csv(Path(data_path, "filled_templates.csv"))

    results_dict = evaluate_texts(text_answer_pairs, "loss")
    json.dump(results_dict, open("cache.json", "w"))
    results_dict = json.load(open("cache.json"))
    pprint(results_dict)
    plot_probs(results_dict)


if __name__ == "__main__":
    format = "(%(threadName)s) %(asctime)s | %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
    logging.info("logger")
    main()
