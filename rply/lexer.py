from .errors import LexingError
from .box import SourcePosition, Token


class Lexer:
    """词法分析器，lex()获取 Token 流"""

    def __init__(self, rules, ignore_rules):
        self.rules = rules
        self.ignore_rules = ignore_rules

    def lex(self, s):
        return LexerStream(self, s)


class LexerStream:
    """词法分析器流，用于生成 Token 流"""

    def __init__(self, lexer, s):
        self.lexer = lexer  # 词法分析器（包含匹配规则）
        self.s = s          # 输入字符串
        self.idx = 0
        self._lineno = 1
        self._colno = 1

    def __iter__(self):
        return self

    def _update_pos(self, match):
        # 更新当前索引到匹配结束位置
        self.idx = match.end
        # 统计匹配范围内的换行符数量，更新行号
        self._lineno += self.s.count("\n", match.start, match.end)
        # 计算最后一个换行符的位置，用于更新列号
        last_nl = self.s.rfind("\n", 0, match.start)
        if last_nl < 0:
            # 匹配范围内没有换行符，列号 = 匹配开始位置 + 1（从1开始计数）
            return match.start + 1
        else:
            # 有换行符，列号 = 匹配开始位置 - 最后换行符位置（新行的偏移量）
            return match.start - last_nl

    def __next__(self):
        # 第一步：跳过忽略规则
        while True:
            if self.idx >= len(self.s):
                raise StopIteration
            for rule in self.lexer.ignore_rules:
                match = rule.matches(self.s, self.idx)
                if match:
                    self._update_pos(match)
                    break
            else:
                break

        # 第二步：匹配有效 Token 规则
        for rule in self.lexer.rules:
            match = rule.matches(self.s, self.idx)
            if match:
                lineno = self._lineno
                self._colno = self._update_pos(match)  # 更新位置，获取新列号
                # 创建源位置对象（包含起始索引、行号、列号）
                source_pos = SourcePosition(match.start, lineno, self._colno)
                # 创建 Token 对象（包含规则名称、匹配的文本、位置信息）
                token = Token(rule.name, self.s[match.start:match.end], source_pos)
                return token
        else:
            raise LexingError(None, SourcePosition(self.idx, self._lineno, self._colno))