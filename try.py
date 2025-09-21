from rply import LexerGenerator, ParserGenerator, BaseBox
from LL1 import Tokenizer, Parser, Interpreter


class Number(BaseBox):
    def __init__(self, value):
        self.value = value

    def eval(self):
        return self.value

class BinaryOp(BaseBox):
    def __init__(self, left, right):
        self.left = left
        self.right = right

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
    lg.add('NUMBER', r'\d+')
    lg.add('PLUS', r'\+')
    lg.add('MINUS', r'-')
    lg.add('MUL', r'\*')
    lg.add('DIV', r'/')
    lg.add('OPEN_PARENS', r'\(')
    lg.add('CLOSE_PARENS', r'\)')
    lg.ignore(r'\s+')
    lexer = lg.build()
    pg = ParserGenerator(
        ['NUMBER', 'OPEN_PARENS', 'CLOSE_PARENS', 'PLUS', 'MINUS', 'MUL', 'DIV', 'POW'],
        precedence=[("left", ["PLUS", "MINUS"]), ("left", ["MUL", "DIV"]), ("right", ["POW"])],
        cache_id="my_parser"
    )

    @pg.production('factor : NUMBER')
    def factor_number(p):
        return Number(int(p[0].get_str()))

    @pg.production('factor : OPEN_PARENS expr CLOSE_PARENS')
    def factor_parens(p):
        return p[1]

    # 新增幂运算的产生式，优先级最高
    @pg.production('power : factor')
    def power_factor(p):
        return p[0]

    @pg.production('power : factor POW power')
    def power_op(p):
        return Pow(p[0], p[2])

    # term 处理乘除法，基于power
    @pg.production('term : power')
    def term_power(p):
        return p[0]

    @pg.production('term : term MUL power')
    @pg.production('term : term DIV power')
    def term_binary_op(p):
        left = p[0]
        right = p[2]
        if p[1].get_type() == "MUL":
            return Mul(left, right)
        elif p[1].get_type() == "DIV":
            return Div(left, right)
        else:
            raise ValueError("Unknown operator")

    @pg.production('expr : term')
    def expr_term(p):
        return p[0]

    @pg.production('expr : expr PLUS term')
    @pg.production('expr : expr MINUS term')
    def expr_binary_op(p):
        left = p[0]
        right = p[2]
        if p[1].get_type() == "PLUS":
            return Add(left, right)
        elif p[1].get_type() == "MINUS":
            return Sub(left, right)
        else:
            raise ValueError("Unknown operator")

    @pg.error
    def error_handler(token):
        raise ValueError(f"Ran into a {token.get_type()} where it wasn't expected at {token.get_source_pos()}")

    parser = pg.build()

    ans = parser.parse(lexer.lex(code)).eval()
    print(ans)

def demo2():
    code = '3*4+5*6+7'
    variables = {}

    tokenizer = Tokenizer(code)
    tokens = list(tokenizer.tokenize())

    parser = Parser(tokens)
    ast = parser.parse_expr()

    interpreter = Interpreter(variables)
    result = interpreter.evaluate(ast)
    print(result)

if __name__ == '__main__':
    demo1(input(">>> "))
