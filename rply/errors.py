class ParserGeneratorError(Exception):
    pass


class ParserGeneratorWarning(Warning):
    pass


class LexingError(Exception):
    def __init__(self, message, source_pos):
        self.message = message
        self.source_pos = source_pos

    def get_source_pos(self):
        return self.source_pos

    def __repr__(self):
        return f'LexingError({self.message!r}, {self.source_pos!r})'


class ParsingError(Exception):
    def __init__(self, message, source_pos):
        self.message = message
        self.source_pos = source_pos

    def get_source_pos(self):
        return self.source_pos

    def __repr__(self):
        return f'ParsingError({self.message!r}, {self.source_pos!r})'