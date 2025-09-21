class ParserState:
    def __init__(self):
        self.count = 0


class RecordingLexer:
    def __init__(self, record, tokens):
        self.tokens = iter(tokens)
        self.record = record

    def next(self):
        s = "None"
        try:
            token = next(self.tokens)
            s = token.get_type()
        finally:
            self.record.append(f"token:{s}")
        return token

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()