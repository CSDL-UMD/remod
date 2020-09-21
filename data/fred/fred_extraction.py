import os
import rdflib
from .fredlib import checkFredSentence
from ratelimiter import RateLimiter
from utils.file import directory_check
from utils.api import get_api_key
from tqdm import tqdm

RL_MINUTE = RateLimiter(max_calls=5, period=60)       # limit to 5 api calls per minute
RL_DAY = RateLimiter(max_calls=1500, period=86400)    # limit to 1500 api calls per day


def generate_rdfs(api_key_file: str, snippets: dict, out_dir: str, api_method: str = 'limited') -> None:
    """Generates a directory of RDF files that have been passed to the FRED API from a set of strings

    Args:
        api_key_file (str): The file path to the api key
        snippets (dict): A dictionary of snippets such that UID: snippet
        out_dir (str): The directory to store generated RDFs within
        api_method ([type], optional): Limited or unlimited api behavior. Defaults to 'limited':str.
    """
    # Check for directory and extract api_key
    directory_check(out_dir)
    api_key = get_api_key(api_key_file)

    # For every snippet, call FRED API
    for uid, snip in tqdm(snippets.items()):
        out_file = out_dir + uid + '.rdf'

        # Based on FRED API Limits with generic API key
        if api_method == 'limited':
            with RL_DAY:
                with RL_MINUTE:
                    try:
                        checkFredSentence(snip, api_key, out_file)
                        print("Successfully parsed ", uid)
                    except Exception as e:
                        print("Failed to parse ", uid, " Snippet: ", snip)
                        print(e)
        # If a special key has been provided by FRED research team...
        elif api_method == 'unlimited':
            try:
                checkFredSentence(snip, api_key, out_file)
                print("Successfully parsed ", uid)
            except Exception as e:
                print("Failed to parse ", uid, " Snippet: ", snip)
                print(e)