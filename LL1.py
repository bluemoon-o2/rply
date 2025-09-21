import re


# 1. 词法分析器：将输入字符串转换为Token
class Tokenizer:
    def __init__(self, code):
        self.code = code
        self.pos = 0
        # 定义Token类型和对应的正则表达式
        self.token_specs = [
            ('NUMBER', r'\d+(\.\d*)?'),  # 数字（整数或小数）
            ('ID', r'[a-zA-Z_][a-zA-Z0-9_]*'),  # 变量名
            ('OP', r'[+\-*/]'),  # 运算符
            ('LPAREN', r'\('),  # 左括号
            ('RPAREN', r'\)'),  # 右括号
            ('SKIP', r'[ \t\n]+'),  # 空白字符（跳过）
            ('MISMATCH', r'.'),  # 不匹配的字符
        ]
        self.token_regex = '|'.join(f'(?P<{name}>{regex})' for name, regex in self.token_specs)

    def tokenize(self):
        for mo in re.finditer(self.token_regex, self.code):
            kind = mo.lastgroup
            value = mo.group()
            if kind == 'SKIP':
                continue
            elif kind == 'MISMATCH':
                raise RuntimeError(f'Unexpected character: {value}')
            else:
                yield kind, value
        yield 'EOF', None


# 2. 语法分析器：将Token转换为抽象语法树（AST）
class Parser:
    def __init__(self, tokens):
        self.tokens = iter(tokens)
        self.advance()  # 初始化当前Token

    def advance(self):
        self.current_token = next(self.tokens, ('EOF', None))

    def parse_factor(self):
        # 处理数字、变量或括号内的表达式
        kind, value = self.current_token
        if kind == 'NUMBER':
            self.advance()
            return 'number', float(value)
        elif kind == 'ID':
            self.advance()
            return 'id', value
        elif kind == 'LPAREN':
            self.advance()
            expr = self.parse_expr()
            if self.current_token[0] != 'RPAREN':
                raise RuntimeError('Missing closing parenthesis')
            self.advance()
            return expr
        else:
            raise RuntimeError(f'Unexpected token: {kind}')

    def parse_term(self):
        # 处理乘法和除法（优先级高于加减）
        node = self.parse_factor()
        while self.current_token[0] in ('OP',) and self.current_token[1] in ('*', '/'):
            op = self.current_token[1]
            self.advance()
            right = self.parse_factor()
            node = ('binary_op', op, node, right)
        return node

    def parse_expr(self):
        # 处理加法和减法
        node = self.parse_term()
        while self.current_token[0] in ('OP',) and self.current_token[1] in ('+', '-'):
            op = self.current_token[1]
            self.advance()
            right = self.parse_term()
            node = ('binary_op', op, node, right)
        return node


# 3. 解释器：执行AST
class Interpreter:
    def __init__(self, variables=None):
        self.variables = variables or {}

    def evaluate(self, node):
        if node[0] == 'number':
            return node[1]
        elif node[0] == 'id':
            return self.variables[node[1]]
        elif node[0] == 'binary_op':
            op, left, right = node[1], node[2], node[3]
            left_val = self.evaluate(left)
            right_val = self.evaluate(right)
            if op == '+':
                return left_val + right_val
            elif op == '-':
                return left_val - right_val
            elif op == '*':
                return left_val * right_val
            elif op == '/':
                return left_val / right_val
        else:
            raise RuntimeError(f'Unknown node type: {node[0]}')


# 示例用法
if __name__ == '__main__':
    # 输入表达式
    code = 'a + 3 * (b - 2)'
    variables = {'a': 5, 'b': 7}  # 变量值

    # 编译+执行流程
    tokenizer = Tokenizer(code)
    tokens = list(tokenizer.tokenize())
    print('Tokens:', tokens)

    parser = Parser(tokens)
    ast = parser.parse_expr()
    print('AST:', ast)

    interpreter = Interpreter(variables)
    result = interpreter.evaluate(ast)
    print('Result:', result)  # 输出：5 + 3 * (7-2) = 20
