# =============================================================================
# Grammar Utils
# =============================================================================
from collections.abc import MutableMapping


class IdentityDict(MutableMapping):
    """一个基于对象id的字典，而不是基于key的字典。"""
    def __init__(self):
        self._contents = {}
        self._keepalive = []

    def __setitem__(self, key, value):
        idx = len(self._keepalive)
        # 保持对key的引用，防止被垃圾回收（否则 id 可能被重用，导致冲突）
        self._keepalive.append(key)
        # 基于id保存
        self._contents[id(key)] = key, value, idx

    def __getitem__(self, key):
        return self._contents[id(key)][1]

    def __delitem__(self, key):
        del self._contents[id(key)]
        for idx, obj in enumerate(self._keepalive):
            if obj is key:
                del self._keepalive[idx]
                break

    def __len__(self):
        return len(self._contents)

    def __iter__(self):
        for key, _, _ in self._contents.values():
            yield key


class Counter:
    """计数器类"""
    def __init__(self):
        self.value = 0

    def incr(self):
        self.value += 1

# =============================================================================
# Graph Utils
# =============================================================================
import sys

LARGE_VALUE = sys.maxsize


def digraph(X, R, FP):
    """
    计算有向图的强连通分量（SCC）。
    :param X: 节点列表。
    :param R: 关系函数，用于获取节点的后继节点。
    :param FP: 节点属性函数，用于获取节点的属性。
    :return: 节点到其强连通分量的映射。
    """
    # 所有节点初始状态为0（未访问）
    N = dict.fromkeys(X, 0)
    stack = []
    F = {}
    for x in X:
        if N[x] == 0:
            traverse(x, N, stack, F, R, FP)
    return F


def traverse(x, N, stack, F, R, FP):
    """
    遍历有向图的节点，计算其强连通分量（SCC）。
    :param x: 当前遍历的节点。
    :param N: 节点访问深度的字典，用于记录节点的访问顺序。
    :param stack: 遍历路径的栈，用于记录遍历的节点顺序。
    :param F: 节点属性的字典，用于记录节点的属性集合。
    :param R: 关系函数，用于获取节点的后继节点。
    :param FP: 节点属性函数，用于获取节点的属性。
    """
    stack.append(x)  # 将当前节点入栈（记录遍历路径）
    d = len(stack)  # 当前深度（栈长度）
    N[x] = d  # 记录节点x的访问深度
    F[x] = FP(x)  # 初始化节点x的属性集合

    rel = R(x)  # 获取x的所有后继节点
    for y in rel:
        if N[y] == 0:  # 若后继节点y未访问，递归遍历y
            traverse(y, N, stack, F, R, FP)
        # 更新当前节点x的深度为自身与y的最小深度（用于强连通分量判断）
        N[x] = min(N[x], N[y])
        # 传播属性：将y的属性合并到x的属性中（去重）
        for a in F.get(y, []):
            if a not in F[x]:
                F[x].append(a)

    if N[x] == d:  # 若当前节点是所在强连通分量的"根"（深度未被后继节点更新）
        # 弹出栈中所有属于当前强连通分量的节点
        N[stack[-1]] = LARGE_VALUE  # 标记为已处理
        F[stack[-1]] = F[x]  # 强连通分量内节点属性统一为根节点的属性
        element = stack.pop()
        while element != x:  # 弹出直到当前节点x
            N[stack[-1]] = LARGE_VALUE
            F[stack[-1]] = F[x]
            element = stack.pop()

# =============================================================================
# Cache Utils
# =============================================================================
import os
import json
import errno
import hashlib
import tempfile

from .grammar import Grammar


def compute_grammar_hash(g: Grammar) -> str:
    hasher = hashlib.sha1()
    hasher.update(g.start.encode())
    hasher.update(json.dumps(sorted(g.terminals)).encode())
    for term, (assoc, level) in sorted(g.precedence.items()):
        hasher.update(term.encode())
        hasher.update(assoc.encode())
        hasher.update(bytes(level))
    for p in g.productions:
        hasher.update(p.name.encode())
        hasher.update(json.dumps(p.precedence).encode())
        hasher.update(json.dumps(p.prod).encode())
    return hasher.hexdigest()


def data_is_valid(g: Grammar, data) -> bool:
    if g.start != data["start"]:
        return False
    if sorted(g.terminals) != data["terminals"]:
        return False
    if sorted(g.precedence) != sorted(data["precedence"]):
        return False
    for key, (assoc, level) in g.precedence.items():
        if data["precedence"][key] != [assoc, level]:
            return False
    if len(g.productions) != len(data["productions"]):
        return False
    for p, (name, prod, (assoc, level)) in zip(g.productions, data["productions"]):
        if p.name != name:
            return False
        if p.prod != prod:
            return False
        if p.precedence != (assoc, level):
            return False
    return True


def _serialize_table(table: 'LRTable'):
    return {
        "lr_action": table.lr_action,
        "lr_goto": table.lr_goto,
        "sr_conflicts": table.sr_conflicts,
        "rr_conflicts": table.rr_conflicts,
        "default_reductions": table.default_reductions,
        "start": table.grammar.start,
        "terminals": sorted(table.grammar.terminals),
        "precedence": table.grammar.precedence,
        "productions": [(p.name, p.prod, p.precedence) for p in table.grammar.productions]
    }


def write_cache(cache_dir, cache_file, table):
    if not os.path.exists(cache_dir):
        try:
            os.makedirs(cache_dir, mode=0o0700)
        except OSError as e:
            if e.errno == errno.EROFS:
                return
            raise

    with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False, mode="w") as f:
        json.dump(_serialize_table(table), f)
    os.rename(f.name, cache_file)