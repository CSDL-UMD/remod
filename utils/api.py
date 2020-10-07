def get_api_key(file_path: str) -> str:
    """
    file_path: filepath to api key
    """
    return open(file_path).read()
