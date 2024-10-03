#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2022 Hector Yee, Bryan Bischoff
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
  Generates a html file of random product recommendations from a json catalog file.
"""
import random
import json
import argparse
from typing import Dict

import pin_util

def read_catalog(catalog: str) -> Dict[str, str]:
    """
      Reads in the product to category catalog.
    """
    with open(catalog, "r") as f:
        data = f.read()
    result = json.loads(data)
    return result

def dump_html(subset, output_html: str) -> None:
    """
      Dumps a subset of items.
    """
    with open(output_html, "w") as f:
        f.write("<HTML>\n")
        f.write("""
        <TABLE><tr>
        <th>Key</th>
        <th>Category</th>
        <th>Image</th>
        </tr>""")
        for item in subset:
            key, category = item
            url = pin_util.key_to_url(key)
            img_url = "<img src=\"%s\">" % url
            out = "<tr><td>%s</td><td>%s</td><td>%s</td></tr>\n" % (key, category, img_url)
            f.write(out)
        f.write("</TABLE></HTML>")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate HTML product recommendations from a JSON catalog.")
    parser.add_argument("--input_file", required=True, help="Input catalog JSON file.")
    parser.add_argument("--output_html", required=True, help="Output HTML file.")
    parser.add_argument("--num_items", type=int, default=10, help="Number of items to recommend.")
    
    args = parser.parse_args()

    catalog = read_catalog(args.input_file)
    catalog = list(catalog.items())
    random.shuffle(catalog)
    dump_html(catalog[:args.num_items], args.output_html)

if __name__ == "__main__":
    main()
    '''bash script
    python random_item_recommender.py --input_file fashion-cat.json --output_html output.html --num_items 10
    '''
