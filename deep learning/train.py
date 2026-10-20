"""Unified entry point. Model-specific arguments are forwarded to its Config."""
import argparse
import importlib
import sys
from eval import MODULES
from utils.training import parse_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=MODULES)
    args, remaining = parser.parse_known_args()
    module = importlib.import_module(MODULES[args.model])
    sys.argv = [sys.argv[0]] + remaining
    return module.main(parse_config(module.Config))


if __name__ == '__main__':
    main()
