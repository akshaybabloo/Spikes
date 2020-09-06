import logging
import sys


def log_it(yes=False):
    """
    Global logging, if set to ``True`` the program runs in verbose mode.

    :param bool yes:
         ``True`` will log the outputs.

    """
    if yes:
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] [%(funcName)s] [%(threadName)-12.12s] "
            "[%(levelname)-5.5s]  %(message)s")
        ch.setFormatter(formatter)
        root.addHandler(ch)
