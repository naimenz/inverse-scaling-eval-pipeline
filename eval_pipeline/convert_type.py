import argparse
import sys
import pandas as pd
from pathlib import Path

def main():
    args = parse_args(sys.argv[1:])
    in_path = Path(args.in_file)
    out_path = Path(args.out_file)
    has_index_col = args.has_index

    convert_type(in_path, out_path, has_index_col)
    
def convert_type(in_path: Path, out_path: Path, has_index_col: bool = False) -> None:
    if in_path.suffix == ".csv":
        if has_index_col:
            df = pd.read_csv(in_path, index_col=0)
        else:
            df = pd.read_csv(in_path)
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
        "in_file",
        type=str,
        help="path of file to read data from",
    )
    parser.add_argument(
        "out_file",
        type=str,
        help="path of file to write data to",
    )

    parser.add_argument(
        "--has-index",
        action="store_true",
        help="whether there's an index col on the input csv",
        required=False,
    )

    args = parser.parse_args(args)
    return args
 

if __name__ == "__main__":
    main()