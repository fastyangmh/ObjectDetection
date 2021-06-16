# import
from ruamel.yaml import safe_load

# def


def load_yaml(file_path):
    with open(file_path, 'r') as f:
        content = safe_load(f)
    return content
