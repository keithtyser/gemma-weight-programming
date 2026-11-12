"""Shared-prefix (tree) packing of an episode's model calls for exact scoring with one forward over unique tokens.

Author: Claude Opus 5.5 (TII trainer track)
Date: 05-November-2026
PURPOSE: An agent episode makes calls k = 0..K-1 whose token sequences S_k = P_k + G_k (prompt + sampled completion)
  share long prefixes but are not strict extensions of each other (the Gemma 4 template re-renders earlier assistant
  turns, so P_k may diverge from P_{k-1} + G_{k-1} at some point). `build_tree` inserts every S_k into a compressed
  token trie: each trie node is a run of tokens whose context (the path from the root) is identical for every call
  passing through it, so its rows are computed ONCE and shared. Packed layout: nodes in DFS pre-order (parents before
  children), one packed row per trie token; every row has an absolute position = its depth on its path.
  For call k, `seq_rows[k][p]` is the packed row of position p of S_k, so token p's logprob is read from the logits of
  row seq_rows[k][p-1] - exactly its own context. The attention plan (ops.AttnPlan.build_tree) makes every node one
  kernel "sequence" whose block table points at its ancestors' K/V pages: the engine's prefix-cache mechanism, which
  the engine track proved bitwise equal to a cache miss (C3, P4 attention fix).
  Reads/writes nothing. Callers: aligned_forward.AlignedGemma4.score_tree, tests/test_trainer_tree.py, bench scripts.
SRP/DRY check: Pass - no trie/shared-prefix packing exists elsewhere in the repo (grpo_data merges rows by alignment,
  which is not exact).
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np


@dataclasses.dataclass
class TreeBatch:
    ids: list[int]  # packed unique tokens
    row_pos: list[int]  # absolute position (depth) of each packed row
    nodes: list[tuple[int, int, int, int]]  # (parent node or -1, row_lo, row_hi, pos_lo) in DFS pre-order
    seq_rows: list[np.ndarray]  # per input sequence: packed row index of each position
    total_input_tokens: int  # sum of len(S_k): what one-row-per-call would process

    @property
    def unique_tokens(self) -> int:
        return len(self.ids)


class _Node:
    __slots__ = ("tokens", "parent", "children")

    def __init__(self, tokens: np.ndarray, parent: "_Node | None"):
        self.tokens = tokens
        self.parent = parent
        self.children: dict[int, _Node] = {}


def _lcp(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    if n == 0:
        return 0
    ne = np.flatnonzero(a[:n] != b[:n])
    return int(ne[0]) if ne.size else n


def _build_trie(sequences: Sequence[Sequence[int]]) -> tuple[_Node, list[_Node]]:
    """Compressed token trie of the sequences; returns (root, leaf node of each sequence)."""
    root = _Node(np.zeros(0, dtype=np.int64), None)
    leaves = []
    for seq in sequences:
        s = np.asarray(seq, dtype=np.int64)
        if s.size == 0:
            raise ValueError("empty sequence")
        node, i = root, 0
        while i < len(s):
            child = node.children.get(int(s[i]))
            if child is None:
                child = _Node(s[i:].copy(), node)
                node.children[int(s[i])] = child
                node, i = child, len(s)
                break
            m = _lcp(child.tokens, s[i:])
            if m < len(child.tokens):  # split the edge at the divergence (or at the end of this sequence)
                mid = _Node(child.tokens[:m].copy(), node)
                node.children[int(s[i])] = mid
                child.tokens = child.tokens[m:]
                child.parent = mid
                mid.children[int(child.tokens[0])] = child
                child = mid
            node, i = child, i + m
        leaves.append(node)
    return root, leaves


def build_tree(sequences: Sequence[Sequence[int]]) -> TreeBatch:
    root, leaves = _build_trie(sequences)
    # DFS pre-order layout
    ids: list[int] = []
    row_pos: list[int] = []
    nodes: list[tuple[int, int, int, int]] = []
    index: dict[int, int] = {}
    pos_lo_of: dict[int, int] = {id(root): 0}
    stack = [(c, root) for c in reversed(list(root.children.values()))]
    while stack:
        n, par = stack.pop()
        p_lo = pos_lo_of[id(par)] + (len(par.tokens) if par is not root else 0)
        pos_lo_of[id(n)] = p_lo
        lo = len(ids)
        ids.extend(int(t) for t in n.tokens)
        row_pos.extend(range(p_lo, p_lo + len(n.tokens)))
        index[id(n)] = len(nodes)
        nodes.append((index[id(par)] if par is not root else -1, lo, len(ids), p_lo))
        stack.extend((c, n) for c in reversed(list(n.children.values())))
    # per sequence: rows of its path
    seq_rows = []
    for seq, leaf in zip(sequences, leaves):
        path = []
        n = leaf
        while n is not root:
            path.append(index[id(n)])
            n = n.parent
        rows = np.concatenate([np.arange(nodes[j][1], nodes[j][2]) for j in reversed(path)])
        if len(rows) < len(seq):
            raise AssertionError("trie path shorter than sequence")
        rows = rows[: len(seq)]
        assert all(ids[r] == t for r, t in zip(rows[:: max(1, len(rows) // 64)], list(seq)[:: max(1, len(rows) // 64)]))
        seq_rows.append(rows)
    return TreeBatch(ids=ids, row_pos=row_pos, nodes=nodes, seq_rows=seq_rows,
                     total_input_tokens=sum(len(s) for s in sequences))


def plan_subtrees(sequences: Sequence[Sequence[int]], budget: int) -> list[list[int]]:
    """Split the calls of an episode into groups whose trie (union of their paths) has <= `budget` unique tokens.

    Every group is scored by its own forward; a group recomputes only the ancestor path of its part of the trie, so
    every sampled token keeps its exact context. Greedy, top-down: a trie node whose whole subtree (plus its ancestor
    path) fits becomes one group; otherwise its children's subtrees are packed first-fit-decreasing into bins of
    capacity budget - (ancestor path + node length) - so the shared ancestor path is paid once per bin, and a child
    too large for any bin is split recursively. Splits therefore happen at the deepest branch points needed, which
    keeps the recomputed prefix small. Raises if a single call is longer than the budget.
    Returns call-index groups (every call in exactly one group, in a deterministic order).
    """
    if not sequences:
        return []
    longest = max(len(x) for x in sequences)
    if longest > budget:
        raise ValueError(f"a single call has {longest} tokens > budget {budget}; cannot split below one call")
    root, leaves = _build_trie(sequences)
    ends: dict[int, list[int]] = {}
    for i, leaf in enumerate(leaves):
        ends.setdefault(id(leaf), []).append(i)
    size: dict[int, int] = {}

    def subtree_size(n: _Node) -> int:  # tokens in n and below (iterative post-order to avoid deep recursion)
        order, stack = [], [n]
        while stack:
            x = stack.pop()
            order.append(x)
            stack.extend(x.children.values())
        for x in reversed(order):
            size[id(x)] = len(x.tokens) + sum(size[id(c)] for c in x.children.values())
        return size[id(n)]

    subtree_size(root)

    def calls_below(n: _Node) -> list[int]:
        out, stack = [], [n]
        while stack:
            x = stack.pop()
            out.extend(ends.get(id(x), []))
            stack.extend(x.children.values())
        return sorted(out)

    groups: list[list[int]] = []

    def split(n: _Node, path_len: int) -> None:
        here = path_len + len(n.tokens)
        if path_len + size[id(n)] <= budget:
            groups.append(calls_below(n))
            return
        cap = budget - here
        bins: list[tuple[int, list[int]]] = []  # (used tokens below n, calls)
        own = sorted(ends.get(id(n), []))  # calls ending exactly at n need only the path
        big = []
        for c in sorted(n.children.values(), key=lambda c: -size[id(c)]):
            sc = size[id(c)]
            if sc > cap:
                big.append(c)
                continue
            for b, (used, cl) in enumerate(bins):
                if used + sc <= cap:
                    bins[b] = (used + sc, cl + calls_below(c))
                    break
            else:
                bins.append((sc, calls_below(c)))
        if own:
            if bins:
                bins[0] = (bins[0][0], own + bins[0][1])
            else:
                bins.append((0, own))
        groups.extend(sorted(cl) for _, cl in bins)
        for c in big:
            split(c, here)

    for c in sorted(root.children.values(), key=lambda c: -size[id(c)]):
        split(c, 0)
    flat = sorted(i for g in groups for i in g)
    if flat != list(range(len(sequences))):
        raise AssertionError("subtree plan does not cover every call exactly once")
    return [g for g in groups if g]
