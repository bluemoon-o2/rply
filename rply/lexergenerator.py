import re
from .lexer import Lexer

try:
    import rpython
    from rpython.rlib.objectmodel import we_are_translated
    from rpython.rlib.rsre import rsre_core
    from rpython.rlib.rsre.rpy import get_code
except ImportError:
    rpython = None
    def we_are_translated():
        return False


class Match:
    """封装匹配索引"""

    _attrs_ = ["start", "end"]

    def __init__(self, start, end):
        self.start = start
        self.end = end


class Rule:
    """封装匹配的名称和正则表达式对象"""

    _attrs_ = ['name', 'flags', '_pattern']

    def __init__(self, name, pattern, flags=0):
        self.name = name
        self.re = re.compile(pattern, flags=flags)
        if rpython:
            self._pattern = get_code(pattern, flags)

    def matches(self, s, pos):
        """
        从位置pos开始解析字符串s
        :return: 如果规则匹配，则返回一个`Match`对象；如果不匹配，则返回None
        """
        if not we_are_translated():
            m = self.re.match(s, pos)
            return Match(*m.span(0)) if m is not None else None
        else:
            assert pos >= 0
            ctx = rsre_core.StrMatchContext(s, pos, len(s))

            matched = rsre_core.match_context(ctx, self._pattern)
            if matched:
                return Match(ctx.match_start, ctx.match_end)
            else:
                return None


class LexerGenerator:
    """
    用于生成词法分析器。

    >>> from rply import LexerGenerator
    >>> lg = LexerGenerator()
    >>> lg.add('NUMBER', r'\d+')
    >>> lg.add('ADD', r'\+')
    >>> lg.ignore(r'\s+')
    >>> import re
    >>> lg.add('ALL', r'.*', flags=re.DOTALL)
    >>> lexer = lg.build()
    >>> iterator = lexer.lex('1 + 1')
    >>> iterator.next()
    Token('NUMBER', '1')
    >>> iterator.next()
    Token('ADD', '+')
    >>> iterator.next()
    Token('NUMBER', '1')
    >>> iterator.next()
    Traceback (most recent call last):
    ...
    StopIteration
    """

    def __init__(self):
        self.rules = []
        self.ignore_rules = []

    def add(self, name, pattern, flags=0):
        """添加匹配规则，第一条优先"""
        self.rules.append(Rule(name, pattern, flags=flags))

    def ignore(self, pattern, flags=0):
        """添加忽略规则，第一条优先"""
        self.ignore_rules.append(Rule("", pattern, flags=flags))

    def build(self):
        """
        返回一个词法分析器实例，该实例提供一个 `lex` 方法
        该方法必须传递一个字符串，返回一个迭代器生成 `Token` 实例。
        """
        return Lexer(self.rules, self.ignore_rules)
