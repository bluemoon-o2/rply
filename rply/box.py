class BaseBox:
    """
    用于封装解析器结果的基类。
    兼容 RPython 要求函数始终返回相同类型的对象。
    """
    _attrs_ = []


class SourcePosition:
    """封装源位置信息（索引，行号，列号）"""

    def __init__(self, idx, lineno, colno):
        self.idx = idx
        self.lineno = lineno
        self.colno = colno

    def __repr__(self):
        return f"SourcePosition(idx={self.idx}, lineno={self.lineno}, colno={self.colno})"


class Token(BaseBox):
    """封装语法分析器生成的令牌"""
    def __init__(self, name, value, source_pos=None):
        self.name = name
        self.value = value
        self.source_pos = source_pos

    def __repr__(self):
        return f"Token({self.name!r}, {self.value!r})"

    def __eq__(self, other):
        if not isinstance(other, Token):
            # 尝试other的比较方法
            return NotImplemented
        return self.name == other.name and self.value == other.value

    def get_type(self):
        return self.name

    def get_source_pos(self):
        return self.source_pos

    def get_str(self):
        return self.value


class BoxInt(BaseBox):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"BoxInt({self.value})"

    def __eq__(self, other):
        return self.value == other.value

    def getint(self):
        return self.value