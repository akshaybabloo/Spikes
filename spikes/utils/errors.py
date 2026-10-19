"""
Exception used through out the Spikes api.
"""

__all__ = [
    "LengthMismatchError",
    "NotEnoughDataError",
    "SplitDataError",
]


class SplitDataError(Exception):
    """
    This exception is raised when the data cannot be split into
    it's appropriate training, testing or validation.
    """

    def __init__(self, message, errors=None):
        super().__init__(message)

        self.errors = errors


class NotEnoughDataError(Exception):
    """
    This exception is raised when the required amount of data is not given.
    """

    def __init__(self, message, errors=None):
        super().__init__(message)

        self.errors = errors


class LengthMismatchError(Exception):
    """
    This exception is raised when the length of sample size is not equal to the length of labels.
    """

    def __init__(self, message, errors=None):
        super().__init__(message)

        self.errors = errors
