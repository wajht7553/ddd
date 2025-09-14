import yaml
import sys
import argparse


def load_config(default_config, cli_args):
    # Load config file if provided
    config = default_config.copy()
    if cli_args.config:
        with open(cli_args.config, "r") as f:
            file_config = yaml.safe_load(f)
            if file_config:
                config.update(file_config)

    # Override config with CLI arguments if provided
    for key in default_config.keys():
        val = getattr(cli_args, key, None)
        if val is not None:
            config[key] = val

    print(f"Configuration: {config}")

    return argparse.Namespace(**config)
