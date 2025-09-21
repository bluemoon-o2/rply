import os
import json
import warnings

from appdirs import AppDirs
from typing import List

from .grammar import Grammar, LRItem
from .parser import LRParser
from .errors import ParserGeneratorError, ParserGeneratorWarning
from .utils import (digraph,
                    Counter, IdentityDict,
                    compute_grammar_hash, data_is_valid, write_cache)



class ParserGenerator:
    """
    解析器生成器（ParserGenerator）表示一组产生式规则，这些规则定义了一系列终结符和非终结符，它们将被替换为一个非终结符，进而可以转换为解析器。
    :param tokens: 标记（非终结符）名称的列表。
    :param precedence: 定义运算顺序以避免歧义的元组列表，由一个定义结合性（左结合、右结合或非结合）的字符串和一个具有相同结合性及优先级级别的标记名称列表组成。
    :param cache_id: 用于指定缓存 ID 的字符串。
    """
    VERSION = 1

    def __init__(self, tokens, precedence=[], cache_id=None):
        self.tokens = tokens
        self.precedence = precedence
        self.cache_id = cache_id
        self.productions = []
        self.error_handler = None

    def production(self, rule, precedence=None):
        parts = rule.split()        # 分割规则字符串为列表（按空格拆分）
        production_name = parts[0]  # 产生式左部（非终结符，如 "expr"）
        if parts[1] != ":":
            raise ParserGeneratorError("Expecting :")  # 校验规则格式（必须包含 ":"）

        body = " ".join(parts[2:])  # 提取 ":" 右侧的所有内容（产生式右部）
        prods = body.split("|")  # 按 "|" 分割多个候选式（同一左部的不同右部）

        def inner(func):
            for production in prods:
                syms = production.split()  # 拆分候选式为符号列表（如 ["expr", "PLUS", "expr"]）
                # 将产生式信息（左部、右部符号、语义函数、优先级）添加到内部列表
                self.productions.append((production_name, syms, func, precedence))
            return func  # 返回原函数

        return inner  # 返回装饰器

    def error(self, func):
        """
        定义解析错误处理函数。
        :param func: 解析错误处理函数，用于处理解析错误。
        :return: 解析错误处理函数。
        """
        self.error_handler = func
        return func

    def build(self):
        g = Grammar(self.tokens)
        # 注册优先级
        for idx, (assoc, terms) in enumerate(self.precedence, 1):
            for term in terms:
                g.set_precedence(term, assoc, idx)
        # 注册产生式
        for prod_name, syms, func, precedence in self.productions:
            g.add_production(prod_name, syms, func, precedence)
        # 设置起始符号
        g.set_start()
        # 检查未使用的终结符和非终结符
        for unused_term in g.unused_terminals:
            warnings.warn(f"Token {unused_term!r} is unused", ParserGeneratorWarning, stacklevel=2)
        for unused_prod in g.unused_non_terminals:
            warnings.warn(f"Production {unused_prod!r} is not reachable", ParserGeneratorWarning, stacklevel=2)
        # 构建 LR 表
        g.build_lr_items()
        # 计算 FIRST 集
        g.compute_first()
        # 计算 FOLLOW 集
        g.compute_follow()

        table = None
        if self.cache_id is not None:
            cache_dir = AppDirs("rply").user_cache_dir
            cache_file = os.path.join(
                cache_dir,
                f"{self.cache_id}-{self.VERSION}-{compute_grammar_hash(g)}.json"
            )
            if os.path.exists(cache_file):
                with open(cache_file) as f:
                    data = json.load(f)
                if data_is_valid(g, data):
                    table = LRTable.from_cache(g, data)

        if table is None:
            table = LRTable.from_grammar(g)
            if self.cache_id is not None:
                write_cache(cache_dir, cache_file, table)

        if table.sr_conflicts:
            warnings.warn(
                f"{len(table.sr_conflicts)} shift/reduce conflict{'s' if len(table.sr_conflicts) > 1 else ''}",
                ParserGeneratorWarning, stacklevel=2)

        if table.rr_conflicts:
            warnings.warn(
                f"{len(table.rr_conflicts)} reduce/reduce conflict{'s' if len(table.rr_conflicts) > 1 else ''}",
                ParserGeneratorWarning, stacklevel=2)

        return LRParser(table, self.error_handler)


class LRTable:
    def __init__(self, grammar, lr_action, lr_goto, default_reductions, sr_conflicts, rr_conflicts):
        self.grammar = grammar
        self.lr_action = lr_action
        self.lr_goto = lr_goto
        self.default_reductions = default_reductions
        self.sr_conflicts = sr_conflicts
        self.rr_conflicts = rr_conflicts

    @classmethod
    def from_cache(cls, grammar, data):
        lr_action = [dict([(str(k), v) for k, v in action.items()]) for action in data["lr_action"]]
        lr_goto = [dict([(str(k), v) for k, v in goto.items()]) for goto in data["lr_goto"]]
        return LRTable(grammar, lr_action, lr_goto, data["default_reductions"], data["sr_conflicts"], data["rr_conflicts"])

    @classmethod
    def from_grammar(cls, grammar):
        c_id_hash = IdentityDict()  # 项目集到ID的映射（缓存）
        goto_cache = {}  # GOTO函数的缓存
        add_count = Counter()  # 计数辅助
        C = cls.lr0_items(grammar, add_count, c_id_hash, goto_cache)  # 生成LR(0)项目集规范族
        cls.add_lalr_lookaheads(grammar, C, add_count, c_id_hash, goto_cache)  # 为LALR(1)计算向前看符号

        lr_action = [None] * len(C)  # 动作表：索引为状态，值为{终结符: 操作}
        lr_goto = [None] * len(C)  # 转移表：索引为状态，值为{非终结符: 目标状态}
        sr_conflicts = []  # 移进-归约冲突记录
        rr_conflicts = []  # 归约-归约冲突记录
        for st, I in enumerate(C):
            st_action = {}
            st_actionp = {}
            st_goto = {}
            for p in I:
                if len(p) == p.lr_index + 1:  # 点已到达产生式末尾（如 A → α·）
                    if p.name == "S'":  # 起始符号的特殊产生式（如 S' → S）
                        # Accept!
                        st_action["$end"] = 0
                        st_actionp["$end"] = p
                    else:
                        # 归约操作：使用向前看符号确定何时归约
                        heads = p.lookaheads[st]  # 向前看符号集合
                        for a in heads:
                            if a in st_action:
                                # 处理冲突（已有动作时）
                                r = st_action[a]
                                if r > 0:
                                    sprec, slevel = grammar.productions[st_actionp[a].number].precedence
                                    rprec, rlevel = grammar.precedence.get(a, ("right", 0))
                                    if (slevel < rlevel) or (slevel == rlevel and rprec == "left"):
                                        st_action[a] = -p.number
                                        st_actionp[a] = p
                                        if not slevel and not rlevel:
                                            sr_conflicts.append((st, repr(a), "reduce"))
                                        grammar.productions[p.number].reduced += 1
                                    elif not (slevel == rlevel and rprec == "non_assoc"):
                                        if not rlevel:
                                            sr_conflicts.append((st, repr(a), "shift"))
                                elif r < 0:
                                    oldp = grammar.productions[-r]
                                    pp = grammar.productions[p.number]
                                    if oldp.number > pp.number:
                                        st_action[a] = -p.number
                                        st_actionp[a] = p
                                        chosenp, rejectp = pp, oldp
                                        grammar.productions[p.number].reduced += 1
                                        grammar.productions[oldp.number].reduced -= 1
                                    else:
                                        chosenp, rejectp = oldp, pp
                                    rr_conflicts.append((st, repr(chosenp), repr(rejectp)))
                                else:
                                    raise ParserGeneratorError("Unknown conflict in state %d" % st)
                            else:
                                # 记录归约操作：用 -p.number 表示归约到产生式 p
                                st_action[a] = -p.number
                                st_actionp[a] = p
                                grammar.productions[p.number].reduced += 1
                else:
                    # 点未到达末尾（如 A → α·β，β非空）
                    i = p.lr_index
                    a = p.prod[i + 1]  # 点后面的符号
                    if a in grammar.terminals:  # 若为终结符，则执行移进
                        # 计算GOTO(I, a)得到目标状态 j
                        g = cls.lr0_goto(I, a, add_count, goto_cache)
                        j = c_id_hash.get(g, -1)
                        if j >= 0:
                            if a in st_action:
                                # 处理冲突（已有动作时）
                                r = st_action[a]
                                if r > 0:
                                    if r != j:
                                        raise ParserGeneratorError("Shift/shift conflict in state %d" % st)
                                elif r < 0:
                                    rprec, rlevel = grammar.productions[st_actionp[a].number].precedence
                                    sprec, slevel = grammar.precedence.get(a, ("right", 0))
                                    if (slevel > rlevel) or (slevel == rlevel and rprec == "right"):
                                        grammar.productions[st_actionp[a].number].reduced -= 1
                                        st_action[a] = j
                                        st_actionp[a] = p
                                        if not rlevel:
                                            sr_conflicts.append((st, repr(a), "shift"))
                                    elif not (slevel == rlevel and rprec == "nonassoc"):
                                        if not slevel and not rlevel:
                                            sr_conflicts.append((st, repr(a), "reduce"))
                                else:
                                    raise ParserGeneratorError("Unknown conflict in state %d" % st)
                            else:
                                # 记录移进操作：用 j 表示移进后转移到状态 j
                                st_action[a] = j
                                st_actionp[a] = p
            nkeys = set()
            for ii in I:
                for s in ii.unique_syms:
                    if s in grammar.non_terminals:
                        nkeys.add(s)
            for n in nkeys:  # n 是非终结符
                # 计算GOTO(I, n)得到目标状态 j
                g = cls.lr0_goto(I, n, add_count, goto_cache)
                j = c_id_hash.get(g, -1)
                if j >= 0:
                    st_goto[n] = j  # 记录非终结符 n 对应的转移状态

            lr_action[st] = st_action
            lr_goto[st] = st_goto

        # 记录默认归约（状态中所有动作都是同一归约操作时）
        default_reductions = [0] * len(lr_action)
        for state, actions in enumerate(lr_action):
            actions = set(actions.values())
            if len(actions) == 1 and next(iter(actions)) < 0:
                default_reductions[state] = next(iter(actions))
        return LRTable(grammar, lr_action, lr_goto, default_reductions, sr_conflicts, rr_conflicts)

    @classmethod
    def lr0_items(cls, grammar, add_count, c_id_hash, goto_cache) -> List[List[LRItem]]:
        """
        构造LR(0)项目集规范族（canonical collection of LR(0) items）

        LR(0)项目是指在产生式右部某个位置带有圆点的产生式，用于表示解析过程中的状态。
        项目集规范族是一组LR(0)项目集，包含了解析过程中所有可能的状态。

        :param grammar: 文法对象，包含所有产生式规则
        :param add_count: 用于计数或调试的辅助参数（具体用途取决于上下文）
        :param c_id_hash: 字典，用于缓存项目集与其ID的映射（避免重复添加）
        :param goto_cache: 缓存goto操作的结果，优化性能
        :return: LR(0)项目集规范族
        """
        # 初始化项目集规范族C，第一个项目集是文法起始产生式的"初始项目"的闭包
        # lr_next表示在产生式右部添加圆点后的项目
        C = [cls.lr0_closure([grammar.productions[0].lr_next], add_count)]
        # 为初始项目集分配ID（索引），存入哈希表
        for i, I in enumerate(C):
            c_id_hash[I] = i
        i = 0
        # 遍历所有项目集，计算它们的goto操作，构建完整的项目集规范族
        while i < len(C):
            I = C[i]
            i += 1
            # 收集当前项目集中所有项目的"圆点后符号"（可能的跳转符号）
            syms = set()
            for ii in I:
                # unique_syms返回项目中圆点后的符号（如果圆点不在末尾）
                syms.update(ii.unique_syms)
            # 对每个可能的符号x，计算goto(I, x)项目集
            for x in syms:
                # 计算从项目集I通过符号x转移后的新项目集（goto操作）
                # 内部会先计算I中所有圆点后为x的项目，再求它们的闭包
                g = cls.lr0_goto(I, x, add_count, goto_cache)
                if not g:
                    continue  # 如果goto结果为空，跳过
                # 如果新项目集g未在规范族中，则添加它
                if g in c_id_hash:
                    continue  # 已存在，无需重复添加
                # 分配新ID（当前规范族长度即为新ID）
                c_id_hash[g] = len(C)
                C.append(g)
        return C

    @classmethod
    def lr0_closure(cls, I: List[LRItem], add_count) -> List[LRItem]:
        """
        计算LR(0)项目集的闭包（closure）

        闭包的定义：对于一个初始项目集I，其闭包是包含以下内容的最小项目集：
        1. I中所有的项目
        2. 若项目集中有项目 A→α·Bβ（圆点后是非终结符B），则B的所有产生式 B→·γ（圆点在最左侧）也必须加入闭包
        3. 重复步骤2，直到没有新的项目可以添加

        :param I: 输入的初始LR(0)项目集（列表）
        :param add_count: 用于跟踪闭包计算过程的计数器（避免重复添加项目）
        :return: 计算完成的LR(0)闭包项目集（列表）
        """
        # 递增计数器，用于标记本次闭包计算中添加的项目
        # 避免同一轮闭包计算中重复处理同一个项目
        add_count.incr()
        # 初始化闭包集合J为输入项目集I的副本
        J = I[:]
        # 标记是否有新项目被添加，用于控制循环
        added = True
        while added:
            added = False
            # 遍历当前闭包中的所有项目
            for j in J:
                # j.lr_after 表示当前项目圆点后的符号（如果是非终结符）
                # 例如对于项目 A→α·Bβ，lr_after 就是 B
                for x in j.lr_after:
                    # 通过比较计数器值，避免重复添加
                    if x.lr0_added == add_count.value:
                        continue
                    # 将非终结符x的所有产生式的"初始项目"（圆点在最左侧）加入闭包
                    # x.lr_next 表示 x→·γ 形式的项目（γ是x产生式的右部）
                    J.append(x.lr_next)
                    # 标记该符号已在本次计算中被处理
                    x.lr0_added = add_count.value
                    added = True
        return J

    @classmethod
    def lr0_goto(cls, I: List[LRItem], x, add_count, goto_cache) -> List[LRItem]:
        """
        计算LR(0)项目集的goto操作结果

        goto操作定义：对于项目集I和符号x，goto(I, x)是所有满足以下条件的项目的闭包：
        1. 项目集中存在项目 A→α·xβ（圆点后是符号x）
        2. 转移后的项目为 A→αx·β（圆点向右移动一位，跳过x）

        :param I: 当前的LR(0)项目集
        :param x: 转移符号（可以是终结符或非终结符）
        :param add_count: 闭包计算中使用的计数器
        :param goto_cache: 缓存goto操作结果的字典，避免重复计算
        :return: 转移后得到的新项目集（经过闭包计算）
        """
        # 从缓存中获取符号x对应的条目，若不存在则初始化一个IdentityDict
        s = goto_cache.setdefault(x, IdentityDict())
        # 存储所有"圆点跨过x后"的项目
        gs = []
        for p in I:
            # p.lr_next表示将当前项目的圆点向右移动一位后的新项目
            # 例如：p是A→α·xβ，则p.lr_next是A→αx·β
            n = p.lr_next
            # 检查移动圆点后的项目是否满足：圆点前的符号是x
            # 即确保我们只收集那些通过x符号转移的项目
            if n and n.lr_before == x:
                # 从缓存中查找该项目是否已处理过
                s1 = s.get(n)
                if not s1:
                    s1 = {}
                    s[n] = s1
                # 将这个转移后的项目加入临时列表
                gs.append(n)
                s = s1  # 更新当前缓存层级
        # 检查缓存中是否已有最终结果（用"$end"标记完整结果）
        g = s.get("$end")
        if not g:
            # 如果有符合条件的转移项目，计算它们的闭包作为goto结果
            if gs:
                g = cls.lr0_closure(gs, add_count)
                s["$end"] = g  # 缓存结果
            else:
                # 如果没有符合条件的项目，缓存空结果
                s["$end"] = gs
        return g

    @classmethod
    def add_lalr_lookaheads(cls, grammar, C: List[List[LRItem]], add_count, c_id_hash, goto_cache):
        """
        为LR(0)项目集规范族中的项目添加LALR(1)向前看符号（lookahead）

        LALR(1)的核心是在LR(0)项目基础上，为每个项目附加一个lookahead符号集，
        用于在解析时决定何时进行归约操作，解决LR(0)中的移进-归约/归约-归约冲突。

        步骤分解：
        1. 计算可空非终结符（能推导出空串的非终结符）
        2. 识别非终结符的转移关系
        3. 计算读入集（read sets）
        4. 计算回溯包含关系和lookahead依赖
        5. 计算跟随集（follow sets）
        6. 为项目添加最终的lookahead符号集

        :param grammar: 文法对象，包含所有产生式规则
        :param C: LR(0)项目集规范族（列表的列表）
        :param add_count: 闭包计算中使用的计数器
        :param c_id_hash: 项目集到ID的映射字典
        :param goto_cache: goto操作结果的缓存字典
        """
        # 1. 计算可空非终结符（nullable non_terminals）
        # 可空非终结符指能够推导出空串ε的非终结符（如A → ε 或 A → B且B可空）
        # 这是计算lookahead符号集的基础
        nullable = cls.compute_nullable_non_terminals(grammar)

        # 2. 找出所有非终结符的转移关系（transitions）
        # 记录项目集中通过非终结符进行的goto转移，用于后续分析状态关系
        trans = cls.find_non_terminal_transitions(grammar, C)

        # 3. 计算读入集（read sets）
        # 读入集表示在某个状态下，通过处理特定符号序列后可能出现的终结符集合
        # 用于确定lookahead符号的来源
        readsets = cls.compute_read_sets(grammar, C, trans, nullable, add_count, c_id_hash, goto_cache)

        # 4. 计算回溯包含关系（lookback includes）和lookahead依赖
        # 确定不同项目集之间的lookahead符号传递关系，即某个项目集的lookahead如何影响其他项目集
        lookback, included = cls.compute_lookback_includes(grammar, C, trans, nullable, add_count, c_id_hash,
                                                           goto_cache)

        # 5. 计算跟随集（follow sets）
        # 跟随集是每个项目最终的lookahead符号集，基于读入集和包含关系推导得出
        follow_sets = cls.compute_follow_sets(trans, readsets, included)

        # 6. 将计算好的follow sets（lookahead符号集）附加到对应的项目上
        cls.add_lookaheads(lookback, follow_sets)

    @classmethod
    def compute_nullable_non_terminals(cls, grammar):
        """
        计算文法中所有可空非终结符（能够推导出空串ε的非终结符）

        可空非终结符的定义：
        1. 若有产生式 A → ε（空产生式），则A是可空的
        2. 若有产生式 A → B₁B₂...Bₙ，且所有Bᵢ都是可空非终结符，则A是可空的
        3. 可空性具有传递性，需迭代计算直到不再有新的可空非终结符被加入

        :param grammar: 文法对象，包含所有产生式规则
        :return: 可空非终结符的集合
        """
        nullable = set()
        # 记录当前可空非终结符的数量，用于判断迭代是否结束
        num_nullable = 0
        # 迭代计算可空非终结符，直到没有新的非终结符被加入
        while True:
            # 遍历所有产生式（跳过索引0的产生式，通常是拓广文法的起始产生式S'→S）
            for p in grammar.productions[1:]:
                # 情况1：如果产生式右部为空（直接推导出空串），则该产生式的左部非终结符可空
                if len(p) == 0:
                    nullable.add(p.name)
                    continue
                # 情况2：检查产生式右部的所有符号是否都是可空非终结符
                for t in p.prod:
                    # 若存在任何符号不可空，则当前产生式的左部非终结符暂时不可空
                    if t not in nullable:
                        break
                else:
                    # 若所有符号都可空（循环正常结束，未触发break），则左部非终结符可空
                    nullable.add(p.name)
            if len(nullable) == num_nullable:
                break
            num_nullable = len(nullable)
        return nullable

    @classmethod
    def find_non_terminal_transitions(cls, grammar, C):
        """
        找出LR(0)项目集规范族中所有通过非终结符进行的状态转移关系

        非终结符转移指：从某个状态（项目集）通过处理一个非终结符，转移到另一个状态。
        这种转移关系用于后续分析lookahead符号的传递路径。

        :param grammar: 文法对象，包含非终结符集合等信息
        :param C: LR(0)项目集规范族（列表，每个元素是一个项目集）
        :return: 非终结符转移关系的列表，每个元素是元组 (源状态索引, 转移非终结符)
        """
        trans = []
        # 遍历每个状态（项目集）及其索引
        for idx, state in enumerate(C):
            # 遍历当前状态中的每个LR(0)项目
            for p in state:
                # p.lr_index 表示项目中圆点的位置（0-based）
                # 若圆点位置小于产生式右部长度-1，说明圆点后还有符号
                # 例如：产生式A→α·β（β非空），则圆点可向右移动
                if p.lr_index < len(p) - 1:
                    # 获取圆点后紧跟的符号（即转移符号）
                    # p.prod 是产生式右部的符号列表
                    # p.lr_index + 1 是圆点后的符号索引
                    transition_symbol = p.prod[p.lr_index + 1]
                    # 构造转移关系元组：(源状态索引, 转移符号)
                    t = (idx, transition_symbol)
                    # 检查转移符号是否为非终结符，且该转移关系未被记录过
                    if transition_symbol in grammar.non_terminals and t not in trans:
                        trans.append(t)
        return trans

    @classmethod
    def compute_read_sets(cls, grammar, C, trans, nullable, add_count, c_id_hash, goto_cache):
        """
        计算LALR(1)分析中的读入集（read sets）

        读入集表示：在非终结符转移路径上，经过一系列符号后可能出现的终结符集合。
        它用于追踪lookahead符号的来源，是计算最终follow集的基础。

        实现方式：通过有向图（digraph）的不动点迭代计算，结合读关系（reads relation）
        和直接读关系（direct read relation）推导出所有状态转移的读入集。

        :param grammar: 文法对象
        :param C: LR(0)项目集规范族
        :param trans: 非终结符转移关系列表（由find_non_terminal_transitions得到）
        :param nullable: 可空非终结符集合
        :param add_count: 闭包计算计数器
        :param c_id_hash: 项目集到ID的映射
        :param goto_cache: goto操作缓存
        :return: 计算得到的读入集（通常是一个字典，映射转移关系到对应的终结符集）
        """
        return digraph(
            # 图的节点：所有非终结符转移关系
            trans,
            # - R: 读关系函数，定义转移间的依赖关系（x依赖y，则y的读入集需传递给x）
            R=lambda x: cls.reads_relation(C, x, nullable, add_count, c_id_hash, goto_cache),
            # - FP: 直接读关系函数：计算转移x的初始读入集（直接可读取的终结符）
            FP=lambda x: cls.dr_relation(grammar, C, x, nullable, add_count, goto_cache)
        )

    @classmethod
    def compute_follow_sets(cls, trans, readsets, include_sets):
        """
        计算LALR(1)分析中的跟随集（follow sets）

        跟随集定义：对于非终结符转移，其跟随集是该转移相关的所有向前看符号（lookahead）的集合，
        用于在解析时决定何时使用该转移对应的产生式进行归约。它是通过读入集（readsets）
        和包含关系（include_sets）推导得出的最终结果。

        实现方式：通过有向图（digraph）的不动点迭代计算，合并读入集和传递包含关系中的符号。

        :param trans: 所有非终结符转移关系的列表
        :param readsets: 读入集字典（转移→直接读入的终结符集）
        :param include_sets: 包含关系字典（转移→它所包含的其他转移列表）
        :return: 计算得到的跟随集字典（转移→对应的lookahead符号集）
        """
        return digraph(
            # 图的节点：所有非终结符转移关系
            trans,
            # 包含关系函数：定义转移间的符号传递依赖（x包含y，则y的跟随集需传递给x）
            R=lambda x: include_sets.get(x, []),
            # 初始符号集函数：转移x的初始跟随集为其读入集readsets[x]
            FP=lambda x: readsets[x],
        )

    @classmethod
    def dr_relation(cls, grammar, C, trans, nullable, add_count, goto_cache):
        """
        计算直接读关系（DR）：非终结符转移路径上直接出现的终结符集合

        直接读关系定义：对于非终结符转移 (state, N)，其直接读入的终结符是指：
        从状态state通过非终结符N转移后，在新项目集中所有项目的圆点后直接出现的终结符。
        这些终结符是lookahead符号的直接来源。

        :param grammar: 文法对象
        :param C: LR(0)项目集规范族
        :param trans: 非终结符转移关系，元组 (源状态索引, 非终结符N)
        :param nullable: 可空非终结符集合（此处未直接使用，保留为兼容参数）
        :param add_count: 闭包计算计数器
        :param goto_cache: goto操作结果缓存
        :return: 直接读入的终结符列表
        """
        state, N = trans
        terms = []
        # 1. 计算从源状态state通过非终结符N转移后的项目集g
        # 即goto(C[state], N)，得到转移后的新项目集
        g = cls.lr0_goto(C[state], N, add_count, goto_cache)
        # 2. 遍历新项目集g中的所有项目，收集圆点后直接出现的终结符
        for p in g:
            # 检查项目中的圆点是否不在产生式右部的末尾（即圆点后还有符号）
            if p.lr_index < len(p) - 1:
                # 获取圆点后紧跟的符号a
                a = p.prod[p.lr_index + 1]
                # 若a是终结符且未被记录，则加入直接读入集合
                if a in grammar.terminals and a not in terms:
                    terms.append(a)
        # 3. 特殊处理：若转移是从初始状态（state=0）通过拓广文法的起始符号
        # 则需添加结束符$end作为直接读入的终结符（表示整个输入的结束）
        # grammar.productions[0]通常是拓广产生式S'→S，其右部第一个符号是S
        if state == 0 and N == grammar.productions[0].prod[0]:
            terms.append("$end")
        return terms

    @classmethod
    def reads_relation(cls, C, trans, empty, add_count, c_id_hash, goto_cache):
        """
        计算读关系（reads relation）：非终结符转移之间的依赖关系

        读关系定义：对于转移trans = (state, N)，若转移后项目集中存在项目
        A→α·Bβ（圆点后是可空非终结符B），则trans依赖于转移(j, B)，其中j是
        转移后项目集的索引。这种依赖意味着trans的读入集需要包含(j, B)的读入集。

        :param C: LR(0)项目集规范族
        :param trans: 非终结符转移关系，元组 (源状态索引, 非终结符N)
        :param empty: 可空非终结符集合（即nullable）
        :param add_count: 闭包计算计数器
        :param c_id_hash: 项目集到ID的映射字典（项目集→状态索引）
        :param goto_cache: goto操作结果缓存
        :return: 依赖的转移关系列表，每个元素是元组 (状态索引j, 非终结符B)
        """
        rel = []
        state, N = trans
        # 1. 计算从源状态state通过非终结符N转移后的项目集g
        g = cls.lr0_goto(C[state], N, add_count, goto_cache)
        # 2. 获取转移后项目集g对应的状态索引j（若不存在则为-1）
        j = c_id_hash.get(g, -1)
        # 3. 遍历项目集g中的所有项目，寻找依赖关系
        for p in g:
            # 检查项目中的圆点是否不在产生式右部的末尾（即圆点后还有符号）
            if p.lr_index < len(p) - 1:
                # 获取圆点后紧跟的符号a
                a = p.prod[p.lr_index + 1]
                # 若a是可空非终结符，则当前转移依赖于从状态j通过a的转移
                if a in empty:
                    rel.append((j, a))
        return rel

    @classmethod
    def compute_lookback_includes(cls, grammar, C, trans, nullable, add_count, c_id_hash, goto_cache):
        """
        计算回溯包含关系（lookback includes）：确定项目集之间的lookahead符号传递关系

        该函数主要完成两项工作：
        1. 计算look_dict：记录转移关系与相关项目的回溯映射（哪些项目需要继承该转移的lookahead）
        2. 计算include_dict：记录转移间的包含关系（哪些转移的lookahead需要传递给当前转移）

        这些关系是LALR(1)合并同心项目集后，确保lookahead符号正确传递的核心机制。

        :param grammar: 文法对象
        :param C: LR(0)项目集规范族
        :param trans: 非终结符转移关系列表
        :param nullable: 可空非终结符集合
        :param add_count: 闭包计算计数器
        :param c_id_hash: 项目集到ID的映射字典
        :param goto_cache: goto操作结果缓存
        :return: 元组 (look_dict, include_dict)，分别为回溯映射和包含关系字典
        """
        # 初始化回溯映射字典：{转移trans: [(状态j, 项目r), ...]}
        # 表示转移trans的lookahead符号需要传递给状态j中的项目r
        look_dict = {}
        # 初始化包含关系字典：{被包含转移i: [包含它的转移列表...]}
        # 表示转移i的lookahead符号需要传递给列表中的转移
        include_dict = {}
        # 将转移列表转换为字典，便于快速判断转移是否存在
        d_trans = dict.fromkeys(trans, 1)
        # 遍历每个非终结符转移关系 (state, N)
        for state, N in trans:
            # 存储当前转移对应的回溯项目列表
            looks = []
            # 存储当前转移需要包含的其他转移列表
            includes = []
            # 遍历源状态state中的所有项目，筛选出左部为N的项目
            for p in C[state]:
                if p.name != N:  # 只关注左部为转移非终结符N的项目
                    continue
                # 记录当前项目的圆点位置
                lr_index = p.lr_index
                # 从源状态开始跟踪状态转移
                j = state
                # 沿着项目p的产生式右部，模拟圆点向右移动的过程
                while lr_index < len(p) - 1:
                    lr_index += 1
                    t = p.prod[lr_index]  # 圆点后的符号
                    # 若当前符号t对应的转移 (j, t) 存在（即t是非终结符）
                    if (j, t) in d_trans:
                        # 检查从当前位置到产生式末尾的符号是否都可空或不存在终结符
                        li = lr_index + 1
                        while li < len(p):
                            # 若遇到终结符，停止检查（不可空）
                            if p.prod[li] in grammar.terminals:
                                break
                            # 若遇到不可空非终结符，停止检查
                            if p.prod[li] not in nullable:
                                break
                            li += 1
                        else:
                            # 若所有符号都可空或到末尾，则当前转移需要包含 (j, t) 的lookahead
                            includes.append((j, t))
                    # 计算通过符号t转移后的项目集g及对应的状态索引j
                    g = cls.lr0_goto(C[j], t, add_count, goto_cache)
                    j = c_id_hash.get(g, -1)  # 更新当前状态为转移后的状态
                # 寻找与项目p对应的回溯项目r（用于确定lookahead的目标项目）
                for r in C[j]:
                    # 筛选条件：项目r与p左部相同、长度相同，且产生式右部匹配
                    if r.name != p.name:
                        continue
                    if len(r) != len(p):
                        continue
                    # 检查产生式右部是否匹配（r的前r.lr_index个符号与p的对应部分相同）
                    i = 0
                    while i < r.lr_index:
                        if r.prod[i] != p.prod[i + 1]:
                            break
                        i += 1
                    else:
                        # 找到匹配的回溯项目，记录 (状态j, 项目r)
                        looks.append((j, r))
            # 更新包含关系字典：被包含的转移指向包含它的转移
            for i in includes:
                include_dict.setdefault(i, []).append((state, N))
            # 记录当前转移对应的回溯项目列表
            look_dict[(state, N)] = looks
        return look_dict, include_dict

    @classmethod
    def add_lookaheads(cls, lookback, follow_set):
        """
        将计算好的向前看符号集（lookahead）添加到对应的LR项目中

        这一步将跟随集（follow_set）中的符号与具体的LR项目关联，
        最终形成完整的LALR(1)项目（包含产生式、圆点位置和lookahead符号集）。

        :param lookback: 回溯映射字典，记录转移关系与相关项目的对应关系
                         结构通常为：{转移trans: [(状态索引, 项目p), ...]}
        :param follow_set: 跟随集字典，记录每个转移对应的lookahead符号集
                           结构通常为：{转移trans: [符号a1, a2, ...]}
        """
        # 遍历回溯映射中的每个转移关系及其关联的项目列表
        for trans, lb in lookback.items():
            # lb是列表，每个元素是元组(状态索引, 项目p)，表示该项目与转移trans相关
            for state, p in lb:
                # 获取当前转移trans对应的跟随集（lookahead符号集）
                f = follow_set.get(trans, [])
                # 为项目p在指定状态下的lookaheads字典初始化条目
                # p.lookaheads是项目p的lookahead存储，结构为：{状态索引: [符号...]}
                heads = p.lookaheads.setdefault(state, [])
                # 将跟随集中的符号添加到项目p的lookahead中（去重）
                for a in f:
                    if a not in heads:
                        heads.append(a)