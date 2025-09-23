from rply import LexerGenerator, ParserGenerator, BaseBox


# ----------------------------
# 1. Define AST nodes
# ----------------------------
class Number(BaseBox):
    def __init__(self, value):
        self.value = value

    def compile(self, context):
        # Generate the instruction to load the number
        return [f"LOAD {self.value}"]


class Variable(BaseBox):
    def __init__(self, name):
        self.name = name

    def compile(self, context):
        # Check if the variable is defined
        if self.name not in context["variables"]:
            raise ValueError(f"Undefined variable: {self.name}")
        # Generate the instruction to load the variable
        return [f"LOAD_VAR {self.name}"]

class Print(BaseBox):
    def __init__(self, expr):
        self.expr = expr

    def compile(self, context):
        expr_instructions = self.expr.compile(context)
        return expr_instructions + ["PRINT"]

class BinaryOp(BaseBox):
    def __init__(self, left, right):
        self.left = left
        self.right = right


class Add(BinaryOp):
    def compile(self, context):
        # Compile the left and right operands
        left_instructions = self.left.compile(context)
        right_instructions = self.right.compile(context)
        return left_instructions + right_instructions + ["ADD"]


class Sub(BinaryOp):
    def compile(self, context):
        left_instructions = self.left.compile(context)
        right_instructions = self.right.compile(context)
        return left_instructions + right_instructions + ["SUB"]


class Mul(BinaryOp):
    def compile(self, context):
        left_instructions = self.left.compile(context)
        right_instructions = self.right.compile(context)
        return left_instructions + right_instructions + ["MUL"]


class Div(BinaryOp):
    def compile(self, context):
        left_instructions = self.left.compile(context)
        right_instructions = self.right.compile(context)
        return left_instructions + right_instructions + ["DIV"]


class Assignment(BaseBox):
    def __init__(self, var_name, expr):
        self.var_name = var_name
        self.expr = expr

    def compile(self, context):
        # Compile the expression
        # Generate the instruction to store the result in the variable
        expr_instructions = self.expr.compile(context)
        # Record that the variable is defined
        context["variables"].add(self.var_name)
        return expr_instructions + [f"STORE {self.var_name}"]


# ----------------------------
# 2. Lexer
# ----------------------------
def build_lexer():
    lg = LexerGenerator()

    lg.add('PRINT', r'print')
    # Keywords
    lg.add('VAR', r'var')
    # Operators
    lg.add('PLUS', r'\+')
    lg.add('MINUS', r'-')
    lg.add('MUL', r'\*')
    lg.add('DIV', r'/')
    lg.add('ASSIGN', r'=')
    # Delimiters
    lg.add('SEMICOLON', r';')
    lg.add('LPAREN', r'\(')
    lg.add('RPAREN', r'\)')
    # Identifier (variable names)
    lg.add('IDENTIFIER', r'[a-zA-Z_][a-zA-Z0-9_]*')
    # Number (supports integers and floating-point numbers)
    lg.add('NUMBER', r'\d+\.?\d*|\.\d+')
    # Ignore spaces and tabs
    lg.ignore(r'\s+')

    return lg.build()


# ----------------------------
# 3. Parser
# ----------------------------
def build_parser():
    # Define all token types
    tokens = [
        'PRINT',
        'VAR', 'IDENTIFIER', 'NUMBER',
        'PLUS', 'MINUS', 'MUL', 'DIV', 'ASSIGN',
        'SEMICOLON', 'LPAREN', 'RPAREN'
    ]

    pg = ParserGenerator(
        tokens,
        precedence=[  # Precedence of operators
            ('left', ['PLUS', 'MINUS']),
            ('left', ['MUL', 'DIV']),
        ]
    )

    # Syntax rules: A program consists of multiple statements.
    @pg.production('program : statement')
    def program_single_statement(p):
        return [p[0]]

    @pg.production('program : program statement')
    def program_multiple_statements(p):
        return p[0] + [p[1]]

    # Syntax rules: A statement can be a variable definition or an assignment.
    @pg.production('statement : var_definition SEMICOLON')
    @pg.production('statement : assignment SEMICOLON')
    def statement(p):
        return p[0]

    @pg.production('statement : PRINT expr SEMICOLON')
    def statement_print(p):
        return Print(p[1])

    # Syntax rules: A variable definition consists of the keyword 'var', a variable name, and an assignment expression.
    @pg.production('var_definition : VAR IDENTIFIER ASSIGN expr')
    def var_definition(p):
        var_name = p[1].get_str()
        expr = p[3]
        return Assignment(var_name, expr)

    # Syntax rules: An assignment statement consists of a variable name, the assignment operator '=', and an expression.
    @pg.production('assignment : IDENTIFIER ASSIGN expr')
    def assignment(p):
        var_name = p[0].get_str()
        expr = p[2]
        return Assignment(var_name, expr)

    # Syntax rules: An expression can be a number, a variable, a parenthesized expression, or a binary operation.
    @pg.production('expr : NUMBER')
    def expr_number(p):
        return Number(float(p[0].get_str()))

    @pg.production('expr : IDENTIFIER')
    def expr_identifier(p):
        return Variable(p[0].get_str())

    @pg.production('expr : LPAREN expr RPAREN')
    def expr_parentheses(p):
        return p[1]

    @pg.production('expr : expr PLUS expr')
    def expr_add(p):
        return Add(p[0], p[2])

    @pg.production('expr : expr MINUS expr')
    def expr_sub(p):
        return Sub(p[0], p[2])

    @pg.production('expr : expr MUL expr')
    def expr_mul(p):
        return Mul(p[0], p[2])

    @pg.production('expr : expr DIV expr')
    def expr_div(p):
        return Div(p[0], p[2])

    # Syntax rules: Error handling.
    @pg.error
    def error_handler(token):
        if token:
            raise ValueError(f"Syntax error: Unexpected token at position {token.get_source_pos()} with type {token.get_type()}")
        else:
            raise ValueError("Syntax error: Expression is incomplete")

    return pg.build()


# ----------------------------
# 4. Compiler Main Function
# ----------------------------
def compile_code(code):
    # 1. Lexical Analysis: Generate Token Stream
    lexer = build_lexer()
    tokens = lexer.lex(code)

    # 2. Syntax Analysis: Generate AST
    parser = build_parser()
    program_ast = parser.parse(tokens)

    # 3. Semantic Analysis and Code Generation
    context = {
        "variables": set()  # Record defined variables
    }
    instructions = []
    for stmt in program_ast:
        instructions.extend(stmt.compile(context))

    return instructions


# ----------------------------
# 5. Virtual Machine (Interpreter)
# ----------------------------
class VirtualMachine:
    def __init__(self):
        self.stack = []
        self.variables = {}

    def run(self, instructions):
        for instr in instructions:
            parts = instr.split()
            op = parts[0]

            if op == "LOAD":
                value = float(parts[1])
                self.stack.append(value)
            elif op == "LOAD_VAR":
                var_name = parts[1]
                if var_name not in self.variables:
                    raise ValueError(f"Undefined variable during execution: {var_name}")
                self.stack.append(self.variables[var_name])
            elif op == "STORE":
                var_name = parts[1]
                value = self.stack.pop()
                self.variables[var_name] = value
            elif op == "ADD":
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(a + b)
            elif op == "SUB":
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(a - b)
            elif op == "MUL":
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(a * b)
            elif op == "DIV":
                b = self.stack.pop()
                a = self.stack.pop()
                if b == 0:
                    raise ZeroDivisionError("Division by zero")
                self.stack.append(a / b)
            elif op == "PRINT":
                value = self.stack.pop()
                print(f"Output: {value}")
            else:
                raise ValueError(f"Unknown instruction: {op}")

# ----------------------------
# 6. Test Compiler
# ----------------------------
if __name__ == "__main__":
    source_code = """
        var a = 10;
        var b = 20;
        var c = (a + b) * 2;
        a = c / 5;
        print a;
        print (b + c);
    """
    try:
        # Compile the source code
        machine_code = compile_code(source_code)
        print("Compilation successful! Generated machine code:")
        for i, instr in enumerate(machine_code, 1):
            print(f"{i}. {instr}")
        print("\nExecuting code...")
        vm = VirtualMachine()
        vm.run(machine_code)
    except Exception as e:
        print(f"Error: {e}")
