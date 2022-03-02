"""I'll be running gpt2 and gpt3 separately to take advantage of some parallelism, and need to combine the results at the end"""
import argparse
import pandas as pd


def main(args: argparse.Namespace):
    df1 = pd.read_csv(args.path_1, index_col=0)
    df2 = pd.read_csv(args.path_2, index_col=0)
    df = pd.concat([df1.set_index("text"), df2.set_index("text")], axis=1).reset_index()
    df.to_csv(args.write_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine data from two runs on the same dataset. NOTE: Assumes both files have an index column.")
    parser.add_argument(
        "--path-1",
        type=str,
        help="The file path (relative or absolute) to the first data file",
        required=True,
    )
    parser.add_argument(
        "--path-2",
        type=str,
        help="The file path (relative or absolute) to the second data file",
        required=True,
    )
    parser.add_argument(
        "--write-path",
        type=str,
        help="The file path (relative or absolute) to write the combined df to",
        required=True,
    )
    args = parser.parse_args()
    main(args)
