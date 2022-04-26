import argparse
import sys
import pandas as pd
from pathlib import Path

def main():
    args = parse_args(sys.argv[1:])
    in_path = Path(args.in_file)
    out_path = Path(args.out_file)

    if in_path.suffix == ".csv":
        df = pd.read_csv(in_path, index_col=0)
    elif in_path.suffix == ".jsonl":
        df = pd.read_json(in_path, lines=True)
    else:
        raise ValueError(f"Unknown suffix {in_path.suffix}")

    if out_path.suffix == ".csv":
        df.to_csv(out_path)
    elif out_path.suffix == ".jsonl":
        df.to_json(out_path, orient="records", lines=True)
    else:
        raise ValueError(f"Unknown suffix {out_path.suffix}")
    

def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Convert from csv to jsonl or vice versa"
    )
    parser.add_argument(
        "--in-file",
        type=str,
        help="path of file to read data from",
        required=True,
    )
    parser.add_argument(
        "--out-file",
        type=str,
        help="path of file to write data to",
        required=False,
    )

    args = parser.parse_args(args)
    return args
 

if __name__ == "__main__":
    main()