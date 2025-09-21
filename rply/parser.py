from .errors import ParsingError
from .box import Token


class LRParser:
    """
    :param lr_table: LR 分析表，存储语法分析所需的核心信息，包括：
                    default_reductions：默认归约规则（在没有输入符号匹配时使用）、
                    lr_action：动作表（决定对于当前状态和输入符号应执行移进还是归约）、
                    lr_goto：转移表（归约后根据非终结符计算下一个状态）、
                    grammar：语法规则集合（包含产生式）、
                    error_handler：错误处理函数，用于处理语法分析过程中遇到的错误
    :param error_handler: 错误处理函数，用于处理解析错误。
    """

    def __init__(self, lr_table, error_handler):
        self.lr_table = lr_table
        self.error_handler = error_handler

    def parse(self, tokenizer, state=None):
        lookahead = None
        lookahead_stack = []  # 前瞻 Token 栈，用于临时存储未处理的 Token
        state_stack = [0]  # 状态栈，用于存储当前分析状态
        sym_stack = [Token("$end", "$end")]  # 符号栈，用于存储已处理的 Token

        current_state = 0
        while True:
            # 默认归约检查：如果当前状态有默认归约，执行归约
            if self.lr_table.default_reductions[current_state]:
                current_state = self._reduce_production(self.lr_table.default_reductions[current_state],
                                                        sym_stack, state_stack, state)
                continue
            # 获取前瞻 Token
            if lookahead is None:
                if lookahead_stack:
                    lookahead = lookahead_stack.pop()
                else:
                    try:
                        # 从词法分析器获取下一个 Token
                        lookahead = next(tokenizer)
                    except StopIteration:
                        lookahead = None
                # 添加结束标记
                if lookahead is None:
                    lookahead = Token("$end", "$end")
            # 查表决定动作
            l_type = lookahead.get_type()
            if l_type in self.lr_table.lr_action[current_state]:
                t = self.lr_table.lr_action[current_state][l_type]
                if t > 0:
                    # 移进（shift）：将当前 Token 压入符号栈，状态切换到新状态
                    state_stack.append(t)
                    current_state = t
                    sym_stack.append(lookahead)
                    lookahead = None
                    continue
                elif t < 0:
                    # 归约（reduce）：根据产生式将符号栈顶部的符号序列归约为非终结符
                    current_state = self._reduce_production(t, sym_stack, state_stack, state)
                    continue
                else:
                    # 接受（accept）：解析成功，返回最终结果
                    n = sym_stack[-1]
                    return n
            else:
                # TODO: actual error handling here
                if self.error_handler is not None:
                    if state is None:
                        self.error_handler(lookahead)
                    else:
                        self.error_handler(state, lookahead)
                    raise AssertionError("For now, error_handler must raise.")
                else:
                    raise ParsingError(None, lookahead.get_source_pos())

    def _reduce_production(self, t, sym_stack, state_stack, state):
        p = self.lr_table.grammar.productions[-t]
        pname = p.name
        p_len = len(p)
        start = len(sym_stack) + (-p_len - 1)
        assert start >= 0
        targ = sym_stack[start + 1:]
        start = len(sym_stack) + (-p_len)
        assert start >= 0
        del sym_stack[start:]
        del state_stack[start:]
        if state is None:
            value = p.func(targ)
        else:
            value = p.func(state, targ)
        sym_stack.append(value)
        current_state = self.lr_table.lr_goto[state_stack[-1]][pname]
        state_stack.append(current_state)
        return current_state