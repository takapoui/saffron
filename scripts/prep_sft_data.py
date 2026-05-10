import argparse
import json
import logging

from saffron.data import SFTPrepConfig, prepare_sft_dataset

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        prep_config = SFTPrepConfig.from_dict(json.load(f)["prep"])

    prepare_sft_dataset(prep_config)
