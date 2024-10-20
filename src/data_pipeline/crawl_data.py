#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fetches images from Pinterest to a local directory.
"""

import time
import json
from typing import FrozenSet
import os
import argparse
import requests
from src import pin_util

def get_keys(input_file: str, max_lines: int) -> FrozenSet[str]:
    """
    Reads in the Shop the look json file and returns a set of keys.
    """
    keys = []
    with open(input_file, "r") as f:
        data = f.readlines()
        count = 0
        for line in data:
            if count >= max_lines:
                break
            count += 1
            row = json.loads(line)
            keys.append(row["product"])
            keys.append(row["scene"])
    return frozenset(keys)

def fetch_image(key: str, output_dir: str, sleep_time: float) -> bool:
    """Fetches an image from Pinterest."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_name = os.path.join(output_dir, f"{key}.jpg")
    if os.path.exists(output_name):
        print(f"{key} already downloaded.")
        return False

    url = pin_util.key_to_url(key)
    print(url)
    while True:
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(output_name, "wb") as f:
                f.write(response.content)
            return True
        except requests.RequestException:
            print("Network error, sleeping and retrying")
            time.sleep(sleep_time)
            sleep_time += 1

def main():
    parser = argparse.ArgumentParser(description="Fetch images from Pinterest")
    parser.add_argument("--input_file", required=True, help="Input json file")
    parser.add_argument("--max_lines", type=int, default=100000, help="Max lines to read")
    parser.add_argument("--sleep_time", type=float, default=10, help="Sleep time in seconds")
    parser.add_argument("--sleep_count", type=int, default=10, help="Sleep every this number of files")
    parser.add_argument("--output_dir", required=True, help="The output directory")
    
    args = parser.parse_args()

    keys = get_keys(args.input_file, args.max_lines)
    total_keys = len(keys)
    keys = sorted(keys)
    print(f"Found {total_keys} unique images to fetch")

    count = 0
    timeout_count = 0
    for key in keys:
        count += 1
        if fetch_image(key, args.output_dir, args.sleep_time):
            timeout_count += 1
            if timeout_count % args.sleep_count == 0:
                time.sleep(args.sleep_time)
        if count % 100 == 0:
            print(f"Fetched {count} images of {total_keys}")

if __name__ == "__main__":
    main()