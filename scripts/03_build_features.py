import argparse
import os

from src.features.build_features import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/features.yaml")
    args = ap.parse_args()

    print(f"[03_build_features] cwd={os.getcwd()}")
    print(f"[03_build_features] config={args.config}")

    out = run(args.config)
    print(f"[03_build_features] done -> {out}")


if __name__ == "__main__":
    main()
