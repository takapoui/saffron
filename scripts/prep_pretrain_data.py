import argparse
import json

from saffron.data import PretrainPrepConfig, load_and_tokenize_dataset
from saffron.helpers import setup_file_logging

if __name__ == "__main__":
    setup_file_logging("prep_pretrain")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--key", type=str, default="prep")
    args = parser.parse_args()
    with open(args.config) as f:
        prep_config = PretrainPrepConfig.from_dict(json.load(f)[args.key])

    load_and_tokenize_dataset(prep_config)
