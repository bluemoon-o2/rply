# RPLY

Welcome to RPLY! A pure Python parser generator.

You can also find the documentation [online](https://rply.readthedocs.io/).
## Examples
With RPLY, you can create a simple compiler like this:

[A simple compiler](./examples/my_compiler.py)
## How to create your own parser:
```python
from rply import LexerGenerator, ParserGenerator, BaseBox

class Number(BaseBox):
    def __init__(self, value):
        self.value = value

    def eval(self):
        return self.value

class Variable(BaseBox):
    variables = {}

    def __init__(self, value):
        self.value = value

    def eval(self):
        if self.value not in Variable.variables:
            raise ValueError(f"Undefined variable: {self.value}")
        return Variable.variables[self.value]

class BinaryOp(BaseBox):
    def __init__(self, left, right):
        self.left = left
        self.right = right

class Assignment(BinaryOp):
    def eval(self):
        value = self.right.eval()
        Variable.variables[self.left.value] = value
        return value

class Add(BinaryOp):
    def eval(self):
        return self.left.eval() + self.right.eval()

class Sub(BinaryOp):
    def eval(self):
        return self.left.eval() - self.right.eval()

class Mul(BinaryOp):
    def eval(self):
        return self.left.eval() * self.right.eval()

class Div(BinaryOp):
    def eval(self):
        return self.left.eval() / self.right.eval()

class Pow(BinaryOp):
    def eval(self):
        return self.left.eval() ** self.right.eval()

def demo1(code: str):
    lg = LexerGenerator()
    lg.add('POW', r'\*\*')
    lg.add('NUMBER', r'\d+\.?\d*|\.\d+')
    lg.add('VARIABLE', r'[a-zA-Z_][a-zA-Z0-9_]*')
    lg.add('ASSIGN', r'=')
    lg.add('PLUS', r'\+')
    lg.add('MINUS', r'-')
    lg.add('MUL', r'\*')
    lg.add('DIV', r'/')
    lg.add('OPEN_PARENS', r'\(')
    lg.add('CLOSE_PARENS', r'\)')
    lg.ignore(r'\s+')
    lexer = lg.build()

    pg = ParserGenerator(
        ['NUMBER', 'VARIABLE', 'ASSIGN', 'OPEN_PARENS', 'CLOSE_PARENS', 'PLUS', 'MINUS', 'MUL', 'DIV', 'POW'],
        precedence=[("right", ["ASSIGN"]), ("left", ["PLUS", "MINUS"]), ("left", ["MUL", "DIV"]), ("right", ["POW"])],
        cache_id="my_parser"
    )

    @pg.production('expr : NUMBER')
    def expr_number(p):
        return Number(float(p[0].get_str()))

    @pg.production('expr : VARIABLE')
    def expr_variable(p):
        return Variable(p[0].get_str())

    @pg.production('expr : OPEN_PARENS expr CLOSE_PARENS')
    def expr_parens(p):
        return p[1]

    @pg.production('expr : expr ASSIGN expr')
    def expr_assignment(p):
        if not isinstance(p[0], Variable):
            raise ValueError("Assignment must have a variable on the left side")
        return Assignment(p[0], p[2])

    @pg.production('expr : expr PLUS expr')
    @pg.production('expr : expr MINUS expr')
    @pg.production('expr : expr MUL expr')
    @pg.production('expr : expr DIV expr')
    @pg.production('expr : expr POW expr')
    def expr_binary_op(p):
        left = p[0]
        right = p[2]
        if p[1].get_type() == "PLUS":
            return Add(left, right)
        elif p[1].get_type() == "MINUS":
            return Sub(left, right)
        elif p[1].get_type() == "MUL":
            return Mul(left, right)
        elif p[1].get_type() == "DIV":
            return Div(left, right)
        elif p[1].get_type() == "POW":
            return Pow(left, right)
        else:
            raise ValueError("Unknown operator")

    @pg.error
    def error_handler(token):
        raise ValueError(f"Ran into a {token.get_type()} where it wasn't expected")

    parser = pg.build()

    ans = parser.parse(lexer.lex(code)).eval()
    print(ans)
```
Then you can do:
```python
if __name__ == '__main__':
    print("Supports floating point numbers, power operator (**), and variable assignment")
    print("Example: a = 2.5; b = a**2 + 3; b")
    print("Input 'quit' to exit")

    while True:
        code = input(">>> ")
        if code.lower() == 'quit':
            break
        # Split input into multiple expressions separated by semicolons
        for expr in code.split(';'):
            expr = expr.strip()
            if expr:
                demo1(expr)
```
You can also substitute your own lexer. A lexer is an object with a `next()`
method that returns either the next token in sequence, or `None` if the token
stream has been exhausted.

## Why do we have the boxes?

In RPython, like other statically typed languages, a variable must have a
specific type, we take advantage of polymorphism to keep values in a box so
that everything is statically typed. You can write whatever boxes you need for
your project.

If you don't intend to use your parser from RPython, and just want a cool pure
Python parser you can ignore all the box stuff and just return whatever you
like from each production method.

## Error handling

By default, when a parsing error is encountered, an `rply.ParsingError` is
raised, it has a method `get_source_pos()`, which returns an
`rply.token.SourcePosition` object.

You may also provide an error handler, which, at the moment, must raise an
exception. It receives the `Token` object that the parser errored on.
```python
pg = ParserGenerator(...)

@pg.error
def error_handler(token):
    raise ValueError(f"Syntax error: Unexpected token at position {token.get_source_pos()} with type {token.get_type()}")
```
## Python compatibility

RPly is tested and known to work under Python 3.8+.
