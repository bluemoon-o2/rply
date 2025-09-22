from rply import LexerGenerator, ParserGenerator, BaseBox
from LL1 import Tokenizer, Parser, Interpreter

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
            raise ValueError(f"未定义的变量: {self.value}")
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
            raise ValueError("赋值语句左侧必须是变量")
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
            return Add(left, right)  # 返回加法结果
        elif p[1].get_type() == "MINUS":
            return Sub(left, right)  # 返回减法结果
        elif p[1].get_type() == "MUL":
            return Mul(left, right)  # 返回乘法结果
        elif p[1].get_type() == "DIV":
            return Div(left, right)  # 返回除法结果
        elif p[1].get_type() == "POW":
            return Pow(left, right)  # 返回幂运算结果
        else:
            raise ValueError("Unknown operator")

    @pg.error
    def error_handler(token):
        raise ValueError(f"Ran into a {token.get_type()} where it wasn't expected")

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
    print("支持浮点数、幂运算(**)和变量赋值")
    print("示例: a = 2.5; b = a**2 + 3; b")
    print("输入'quit'退出")

    while True:
        code = input(">>> ")
        if code.lower() == 'quit':
            break
        # 支持分号分隔多个表达式
        for expr in code.split(';'):
            expr = expr.strip()
            if expr:
                demo1(expr)
