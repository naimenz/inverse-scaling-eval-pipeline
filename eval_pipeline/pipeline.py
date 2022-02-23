from __future__ import annotations
from eval_pipeline.gpt2 import evaluate_gpt2_texts, GPT2Size
from eval_pipeline.gpt3 import evaluate_gpt3_texts, GPT3Size
from eval_pipeline.utils import size_dict
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import logging
from pprint import pprint


def evaluate_texts(
    texts,
    gpt2_sizes: list[GPT2Size] = ["gpt2", "gpt2-medium", "gpt2-large"],
    gpt3_sizes: list[GPT3Size] = ["ada", "babbage", "curie", "davinci"],
) -> dict[str, dict[str, float]]:

    gpt2_dict = evaluate_gpt2_texts(texts, gpt2_sizes)
    gpt3_dict = evaluate_gpt3_texts(texts, gpt3_sizes)
    # TODO: get multithreading working properly
    # with ThreadPoolExecutor(max_workers=2) as executor:
    #     gpt2_dict = executor.submit(evaluate_gpt2_texts, texts, gpt2_sizes)
    #     gpt3_dict = executor.submit(evaluate_gpt3_texts, texts, gpt3_sizes)
    # combined_dict = {**gpt2_dict.result(), **gpt3_dict.result()}
    print("gpt2")
    pprint(gpt2_dict)
    print("gpt3")
    pprint(gpt3_dict)
    combined_dict = {text: {**gpt2_dict[text], **gpt3_dict[text]} for text in gpt2_dict.keys()}
    print("combined")
    pprint(combined_dict)
    return combined_dict


def plot_probs(prob_dicts: dict[str, dict[str, float]]) -> None:
    fig = plt.figure(figsize=(20, 10))
    for text in prob_dicts.keys():
        prob_dict = prob_dicts[text]
        xy_pairs = [(size_dict[size], prob) for size, prob in prob_dict.items()]
        xs, ys = zip(*sorted(xy_pairs, key=lambda pair: pair[0]))
        plt.plot(xs, ys, label=text)
    labels, ticks = zip(*size_dict.items())
    # labels = ["A", "B", "C"]
    # ticks = [1e9, 1e10, 1e11]
    plt.xscale("log")
    plt.xticks(ticks, labels, rotation=90)
    plt.legend()
    plt.savefig("fig.png")
    plt.show()


def main():
    texts = [
        ("You just lost at roulette. Should you play again?", " No"),
        ("You just won at roulette. Should you play again?", " No")
    ]
    import json

    positive_prob_dict = evaluate_texts(texts)
    json.dump(positive_prob_dict, open("cache.json",'w'))
    positive_prob_dict = json.load(open("cache.json"))
    plot_probs(positive_prob_dict)


if __name__ == "__main__":
    format = "(%(threadName)s) %(asctime)s | %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
    logging.info("logger")
    main()
