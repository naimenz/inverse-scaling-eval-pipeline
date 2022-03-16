from __future__ import annotations
import pandas as pd
from pathlib import Path


def main():
    base_path = Path(
        "/home/ian/code/lm_internship/eval-pipeline/raw_data/imdb/aclImdb/train"
    )
    pos_dir = Path(base_path, "pos")
    neg_dir = Path(base_path, "neg")

    pos_paths = [path for path in pos_dir.glob("*.txt")]
    neg_paths = [path for path in neg_dir.glob("*.txt")]

    def df_from_paths(paths: list[Path], classes, answer_index) -> pd.DataFrame:
        rows = []
        for file in paths:
            with file.open() as f:
                review = f.read()
            # the dataset has html style line breaks rather than newlines
            review = review.replace("<br />", "\n")
            rows.append(
                {
                    "prompt": review,
                    "classes": classes,
                    "answer_index": answer_index,
                }
            )
        df = pd.DataFrame.from_dict(rows)
        return df
    classes = "[' positive', ' negative']"
    pos_df = df_from_paths(pos_paths, classes, 0)
    neg_df = df_from_paths(neg_paths, classes, 1)
    # df = pos_df.join(neg_df)
    df = pd.concat((pos_df, neg_df))
    print(df.head())
    print(df.tail())
    print(df.info())
    df.to_csv("/home/ian/code/lm_internship/eval-pipeline/data/imdb_train.csv")


if __name__ == "__main__":
    main()
