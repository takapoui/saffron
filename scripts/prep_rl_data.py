import argparse
import json

from saffron.data import RLPrepConfig, prepare_rl_dataset
from saffron.helpers import setup_file_logging

if __name__ == "__main__":
    setup_file_logging("prep_rl")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--key", type=str, default="prep")
    args = parser.parse_args()
    with open(args.config) as f:
        prep_config = RLPrepConfig.from_dict(json.load(f)[args.key])

    prepare_rl_dataset(prep_config)
