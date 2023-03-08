class InvalidArgumentException(Exception):
    def __init__(self, message):
        super().__init__(self.message)


class InvalidGeneratorArgumentException(Exception):
    def __init__(self, message):
        super().__init__(message)


