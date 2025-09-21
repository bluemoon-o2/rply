from .errors import ParserGeneratorError


def rightmost_terminal(symbols, terminals):
    for sym in reversed(symbols):
        if sym in terminals:
            return sym
    return None


class Grammar:
    def __init__(self, terminals):
        self.productions = [None]  # 产生式列表（索引从1开始）
        self.prod_names = {}  # 按非终结符分组的产生式（键：非终结符，值：产生式列表）
        self.terminals = dict((t, []) for t in terminals)  # 终结符及其关联的产生式编号
        self.terminals["error"] = []  # 特殊终结符：错误处理
        self.non_terminals = {}  # 非终结符及其关联的产生式编号
        self.first = {}  # First集（键：符号，值：First集元素列表）
        self.follow = {}  # Follow集（键：非终结符，值：Follow集元素列表）
        self.precedence = {}  # 优先级字典（键：终结符，值：(结合性, 优先级等级)）
        self.start = None  # 文法起始符号

    def add_production(self, prod_name, syms, func, precedence):
        # 检查产生式名称合法性（不能与终结符重名）
        if prod_name in self.terminals:
            raise ParserGeneratorError(f"Illegal rule name {prod_name!r}")
        # 处理优先级（未指定时用最右侧终结符的优先级）
        if precedence is None:
            precedence_name = rightmost_terminal(syms, self.terminals)
            prod_precedence = self.precedence.get(precedence_name, ("right", 0))
        else:
            try:
                prod_precedence = self.precedence[precedence]
            except KeyError:
                raise ParserGeneratorError(f"Precedence {precedence!r} doesn't exist")

        p_idx = len(self.productions)
        self.non_terminals.setdefault(prod_name, [])

        for t in syms:
            if t in self.terminals:
                self.terminals[t].append(p_idx)
            else:
                self.non_terminals.setdefault(t, []).append(p_idx)
        # 创建Production对象并添加到内部存储
        p = Production(p_idx, prod_name, syms, prod_precedence, func)
        self.productions.append(p)
        self.prod_names.setdefault(prod_name, []).append(p)

    def set_precedence(self, term, assoc, idx):
        # 检查合法性并存储（结合性必须是left/right/non_assoc）
        if term in self.precedence:
            raise ParserGeneratorError(f"Precedence already specified for {term!r}")
        if assoc not in ["left", "right", "non_assoc"]:
            raise ParserGeneratorError(f"Precedence must be one of left, right, non_assoc; not {assoc!r}")
        self.precedence[term] = (assoc, idx)

    def set_start(self):
        # 将第一个产生式的左部设为起始符号S
        # 自动添加一个新的起始产生式S' -> S（用于LR分析）
        start = self.productions[1].name
        self.productions[0] = Production(0, "S'", [start], ("right", 0), None)
        self.non_terminals[start].append(0)
        self.start = start

    @property
    def unused_terminals(self):
        return [t for t, prods in self.terminals.items() if not prods and t != "error"]

    @property
    def unused_non_terminals(self):
        return [p for p, prods in self.non_terminals.items() if not prods]

    def build_lr_items(self):
        """遍历产生式列表并构建一套完整的 LR 项"""
        for p in self.productions:
            i = 0  # 点的位置索引（初始在最左侧）
            lr_items = []  # 存储当前产生式的所有 LR 项
            last_lri = p   # 表头为产生式

            while True:
                if i > len(p):
                    lri = None  # 点超出产生式长度，结束循环
                else:
                    # 计算点左侧的符号（before）和右侧可能的符号（after）
                    try:
                        before = p.prod[i - 1]  # 点左侧的符号（如 X 在 X·Y Z 中）
                    except IndexError:
                        before = None  # 点在最左侧时，无左侧符号

                    try:
                        after = self.prod_names[p.prod[i]]  # 点右侧符号的产生式（用于后续分析）
                    except (IndexError, KeyError):
                        after = []  # 点在最右侧时，无右侧符号

                    # 创建 LR 项对象
                    lri = LRItem(p, i, before, after)

                # 动态添加属性实现链表
                last_lri.lr_next = lri
                if lri is None:
                    break
                lr_items.append(lri)  # 保存当前 LR 项
                last_lri = lri
                i += 1

            p.lr_items = lr_items  # 将生成的 LR 表绑定到产生式

    def _first(self, beta):
        result = []
        for x in beta:
            x_produces_empty = False  # 标记当前符号是否能推导出空串
            for f in self.first[x]:  # 遍历当前符号x的First集
                if f == "<empty>":
                    x_produces_empty = True
                else:
                    if f not in result:
                        result.append(f)  # 将非空终结符加入结果
            if not x_produces_empty:
                break  # 若x不能推导出空串，停止处理后续符号
        else:
            # 若所有符号都能推导出空串（循环正常结束，未被break）
            result.append("<empty>")
        return result

    def compute_first(self):
        # 初始化终结符的First集
        for t in self.terminals:
            self.first[t] = [t]
        self.first["$end"] = ["$end"]  # 输入结束标记的First集
        # 初始化非终结符的First集为空
        for n in self.non_terminals:
            self.first[n] = []

        changed = True
        while changed:  # 迭代更新，直到没有新元素加入
            changed = False
            for n in self.non_terminals:
                for p in self.prod_names[n]:  # 遍历n的所有产生式
                    # 计算产生式右部的First集
                    for f in self._first(p.prod):
                        if f not in self.first[n]:
                            self.first[n].append(f)  # 加入新元素
                            changed = True  # 标记有更新，需要继续迭代

    def compute_follow(self):
        # 初始化所有非终结符的Follow集为空列表
        for k in self.non_terminals:
            self.follow[k] = []
        # 起始符号的Follow集初始包含输入结束标记 $end
        self.follow[self.start] = ["$end"]

        added = True
        while added:
            added = False
            for p in self.productions[1:]:  # 遍历所有产生式（跳过第一个可能的空产生式）
                for i, B in enumerate(p.prod):  # 遍历产生式右部的每个符号 B
                    if B in self.non_terminals:  # 仅处理非终结符 B
                        # 步骤1：计算 B 后面的符号序列（β = p.prod[i+1:]）的First集
                        fst = self._first(p.prod[i + 1:])
                        has_empty = False  # 标记 β 是否能推导出空串
                        # 将 β 的First集中非空终结符加入 Follow(B)
                        for f in fst:
                            if f != "<empty>" and f not in self.follow[B]:
                                self.follow[B].append(f)
                                added = True  # 标记有更新
                            if f == "<empty>":
                                has_empty = True  # β 能推导出空串
                        # 步骤2：若 β 能推导出空串，或 B 是产生式右部的最后一个符号（β 为空）
                        # 则将产生式左部 A（p.name）的Follow集元素加入 Follow(B)
                        if has_empty or i == (len(p.prod) - 1):
                            for f in self.follow[p.name]:
                                if f not in self.follow[B]:
                                    self.follow[B].append(f)
                                    added = True  # 标记有更新


class Production:
    def __init__(self, idx, name, syms, precedence, func):
        self.name = name
        self.prod = syms
        self.number = idx
        self.func = func
        self.precedence = precedence
        self.unique_syms = set(syms)
        self.lr_items = []   # LR 表
        self.lr_next = None  # 下一个 LR 项
        self.lr0_added = 0   # 用于LR(0)项目集规范族的计算，记录当前项目是否已被添加
        self.reduced = 0

    def __repr__(self):
        return f"Production({self.name} -> {' '.join(self.prod)})"

    def __len__(self):
        return len(self.prod)


class LRItem:
    def __init__(self, p, n, before, after):
        self.name = p.name
        self.prod = p.prod[:]
        self.prod.insert(n, ".")
        self.number = p.number
        self.lr_index = n
        self.lookaheads = {}  # 项目的lookahead符号集，键为状态，值为lookahead符号列表
        self.unique_syms = p.unique_syms
        self.lr_before = before
        self.lr_after = after

    def __repr__(self):
        return f"LRItem({self.name} -> {' '.join(self.prod)})"

    def __len__(self):
        return len(self.prod)
