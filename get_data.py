"""
Task 1
Downloads GREC Data into a specified directory
"""

import argparse
import os
import pathlib
from shutil import copyfile
import requests

import config
from utils.file import directory_check

GREC_URLS = [
    "https://github.com/mjsumpter/google-relation-extraction-corpus-augmented/raw/master/dob_augment-200526.json",
    "https://github.com/mjsumpter/google-relation-extraction-corpus-augmented/raw/master/education_augment-200526.json",
    "https://github.com/mjsumpter/google-relation-extraction-corpus-augmented/raw/master/institution_augment-200526.json",
    "https://github.com/mjsumpter/google-relation-extraction-corpus-augmented/raw/master/pob_augment-200526.json",
    "https://github.com/mjsumpter/google-relation-extraction-corpus-augmented/raw/master/pod_augment-200526.json",
]


def arg_parse(arg_list=None):
    parser = argparse.ArgumentParser(description="Download Augmented GREC Corpus")
    # Save Directory
    parser.add_argument(
        "--output-directory",
        "-out",
        dest="output_dir",
        help=f"Output Directory Path, default {config.JSON_DIR}",
        type=str,
        default=config.JSON_DIR,
    )

    # Parses and returns args
    if arg_list:
        return parser.parse_args(args=arg_list)
    else:
        return parser.parse_args()


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download_file(url, destination):
    session = requests.Session()
    response = session.get(url)
    destination += "/" + pathlib.Path(url).name
    save_response_content(response, destination)


args = arg_parse()
dir = args.output_dir

directory_check(dir)

for url in GREC_URLS:
    print(f"Downloading { pathlib.Path(url).name } ...")
    download_file(url, dir)

print(f"Copying ./data/cr.json to {dir}")
copyfile('./data/cr.json', (dir + '/cr.json'))