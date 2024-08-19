import json


def save_json(data, filename, indent=None):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)
