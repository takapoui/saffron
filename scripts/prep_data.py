import argparse
import json
import logging

from saffron.config import PrepConfig
from saffron.dataloader import load_and_tokenize_dataset

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        prep_config = PrepConfig.from_dict(json.load(f)["data"])

    load_and_tokenize_dataset(prep_config)
