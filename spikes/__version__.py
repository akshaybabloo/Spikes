

from importlib.metadata import version


def get_versions():
    return {'version': version('pyspikes')}
