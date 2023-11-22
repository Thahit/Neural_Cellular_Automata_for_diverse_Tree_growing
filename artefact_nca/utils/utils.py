import os


def makedirs(path):
    path = path.replace('\\', '/')
    if not os.path.exists(path):
        os.makedirs(path)
    return path
