import argparse
from pathlib import Path

import tiktoken

from saffron.dataloader import load_and_tokenize_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="stas/openwebtext-10k")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--shard_size", type=int, default=100_000)
    args = parser.parse_args()
    load_and_tokenize_dataset(
        dataset=args.dataset,
        name=args.name,
        shard_size=args.shard_size,
        dest_dir=Path(__file__).parent.parent / "data" / args.dataset.replace("/", "_"),
        enc=tiktoken.get_encoding("gpt2"),
        dataset_split="train",
        num_validation_shards=1,
    )
