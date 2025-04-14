import os
from collections import defaultdict


def get_image(path):
    with open(path, "rb") as f:
        image = f.read()  # Read the file as binary data
        return image
