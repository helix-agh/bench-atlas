"""Random function tree generator (Tian et al. 2020), ported from doe2vec.

The tree generator, the expression builder and the acceptance filter are a port of
``nikivanstein/doe2vec`` (van Stein et al. 2023; ``modulesRandFunc`` and
``generateData``, commit 4d498c2), which ports the MATLAB generator of Tian et al.
2020 to Python. The port keeps doe2vec's logic and its order of random draws; the
global ``random`` and ``np.random`` states are replaced by generators passed in.
Three operands are fixed (docs/random-trees.md), so the same seed gives doe2vec's
expression string up to those operands.

doe2vec is distributed under the MIT License:

    Copyright (c) 2022 Bas van Stein

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Node values (operands 1-7, operators 11-36):

    1  real number in [1, 10]     11 x + y     21 -x          29 cos(2 pi x)
    2  decision vector x          12 x - y     22 1 / x       30 log|x|
    3  first variable x1          13 x * y     23 10 x        31 exp(x)
    4  (x2, ..., xd, 0)           14 x / y     24 x^2         32 sum(x)
    5  rotated vector x R                      25 sqrt|x|     33 mean(x)
    6  index vector (1, ..., d)                26 |x|         34 cumsum(x)
    7  noise in [1, 1.1]                       27 round(x)    35 prod(x)
                                               28 sin(2 pi x) 36 max(x)
"""

from __future__ import annotations

import functools
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import qmc

from bench_atlas.problems import Problem, ProblemId

__all__ = ["random_trees"]

# Number of random growth steps of a tree, as doe2vec calls generate_tree(6, 16).
MIN_LENGTH = 6
MAX_LENGTH = 16


def _dot(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    # einsum, unlike np.dot (BLAS), gives each row bitwise independently of the batch.
    return np.einsum("ij,jk->ik", x, m)


# Names an expression may use; no builtins.
_NAMESPACE = {"__builtins__": {}, "np": np, "abs": np.abs, "len": len, "range": range, "dot": _dot}


class _Node:
    def __init__(self, value: int, left: Any = None, right: Any = None) -> None:
        self.value = value
        self.left = left
        self.right = right

    def type(self) -> int:
        # 0 operand, 1 unary operator, 2 binary operator
        if self.left is None:
            return 0
        return 1 if self.right is None else 2

    def is_constant(self) -> bool:
        return self.value <= 1

    def is_scalar(self) -> bool:
        return self.value in (1, 3, 7) or self.value <= 1

    def is_binary(self) -> bool:
        return self.value in (11, 12, 13, 14)

    def is_vector(self) -> bool:
        return self.value in (32, 33, 34, 35, 36)


def _cumulative(weights: list[int]) -> np.ndarray:
    p = np.cumsum(weights)
    return p / np.max(p)


_P_OPERAND = _cumulative([30, 5, 1, 1, 1, 1, 0])
_OPERANDS = np.arange(1, 8)
_P_OPERATOR = _cumulative([15, 15, 10, 10, 2, 2, 2, 5, 5, 2, 2, 3, 3, 3, 3, 3, 3, 1, 1, 1])
_OPERATORS = np.concatenate((np.arange(11, 15), np.arange(21, 37)))


def _tree(rng: random.Random, min_length: int, max_length: int) -> _Node:
    """doe2vec ``generate_tree``."""
    tree = _Node(33, _Node(2))
    for _ in range(rng.randint(min_length, max_length)):
        p = tree
        while p.type() > 0:
            p = p.left if p.type() == 1 or rng.random() < 0.5 else p.right
        operator = int(_OPERATORS[np.argwhere(rng.random() <= _P_OPERATOR)[0][0]])
        if operator <= 20:
            operand = int(_OPERANDS[np.argwhere(rng.random() <= _P_OPERAND)[0][0]])
            if rng.random() < 0.5:
                p.left, p.right = _Node(p.value), _Node(operand)
            else:
                p.left, p.right = _Node(operand), _Node(p.value)
        else:
            p.left = _Node(p.value)
        p.value = operator
    tree = _inject(tree, rng)
    _clean_unary(tree)
    _clean_binary(tree)
    _clean_unary(tree)
    _clean_binary(tree)
    return tree


def _inject(tree: _Node, rng: random.Random) -> _Node:
    """doe2vec ``injection``: add a difficulty with probability 0.45."""
    r = rng.random()
    if r < 0.05:  # noise
        return _Node(13, tree, _Node(7))
    if r < 0.1:  # flat
        return _Node(27, tree)
    if r < 0.2:  # multimodal
        return _Node(11, tree, _Node(28, tree))
    if r < 0.25:  # highly multimodal
        return _Node(11, tree, _Node(23, _Node(28, tree)))
    for bound, kind in ((0.3, 1), (0.35, 2), (0.4, 3), (0.45, 4)):  # variable linkages
        if r < bound:
            return _inject_linkage(tree, kind)
    return tree


def _inject_linkage(tree: Any, kind: int) -> Any:
    """doe2vec ``injection2``: replace every decision vector x."""
    if isinstance(tree, _Node):
        if tree.value == 2:
            return {
                1: _Node(12, _Node(2), _Node(3)),
                2: _Node(12, _Node(2), _Node(4)),
                3: _Node(5),
                4: _Node(13, _Node(6), _Node(2)),
            }[kind]
        tree.left = _inject_linkage(tree.left, kind)
        tree.right = _inject_linkage(tree.right, kind)
    return tree


def _clean_unary(tree: _Node) -> bool:
    """doe2vec ``cleaning1``; returns whether the subtree is a scalar."""
    if tree.type() == 0:
        return tree.is_scalar()
    if tree.type() == 2:
        left = _clean_unary(tree.left)
        right = _clean_unary(tree.right)
        return left and right
    scalar = _clean_unary(tree.left)
    child = tree.left
    if scalar and tree.is_vector():
        # vector operator of a scalar: replace the node with its child
        tree.value, tree.left, tree.right = child.value, child.left, child.right
    elif tree.value == 26 and child.value in (25, 26, 30):
        # abs of abs, log or sqrt: drop the abs
        tree.value, tree.left = child.value, child.left
    elif tree.value in (25, 26, 30) and child.value == 26:
        # abs, log or sqrt of abs: drop the inner abs
        tree.left = child.left
    elif (tree.value, child.value) in ((21, 21), (22, 22), (24, 25), (25, 24), (30, 31), (31, 30)):
        # an operator and its inverse: replace the node with the grandchild
        grandchild = child.left
        tree.value, tree.left, tree.right = grandchild.value, grandchild.left, grandchild.right
    elif (tree.value, child.value) in ((21, 12), (22, 14)):
        # -(a - b) = b - a, 1 / (a / b) = b / a
        tree.value, tree.left, tree.right = child.value, child.right, child.left
    return scalar or (tree.is_vector() and tree.value != 34)


def _clean_binary(tree: _Node) -> bool:
    """doe2vec ``cleaning2``; returns whether the subtree is a constant."""
    if tree.type() == 0:
        return tree.is_constant()
    if tree.type() == 1:
        return _clean_binary(tree.left)
    left_constant = _clean_binary(tree.left)
    right_constant = _clean_binary(tree.right)
    if left_constant and right_constant:
        tree.value, tree.left, tree.right = 1, None, None
        return True
    left, right = tree.left, tree.right
    if right.value == 21 and tree.value in (11, 12):
        # a + -b = a - b, a - -b = a + b
        tree.value, tree.right = 23 - tree.value, right.left
    elif right.value == 22 and tree.value in (13, 14):
        # a * (1/b) = a / b, a / (1/b) = a * b
        tree.value, tree.right = 27 - tree.value, right.left
    elif right.value in (21, 22):
        pass
    elif left.value == 21 and tree.value == 11:
        # -a + b = b - a
        tree.value, tree.left, tree.right = 12, right, left.left
    elif left.value == 22 and tree.value == 13:
        # (1/a) * b = b / a
        tree.value, tree.left, tree.right = 14, right, left.left
    elif left.is_binary() and right_constant:
        if _same_group(tree.value, left.value) and (
            left.left.is_constant() and left.right.is_constant()
        ):
            tree.value, tree.left, tree.right = left.value, left.left, left.right
    elif left_constant and right.is_binary():
        if _same_group(tree.value, right.value):
            _fold_constant(tree)
    elif (
        left.is_binary()
        and right.is_binary()
        and _same_group(left.value, tree.value, right.value)
        and (left.left.is_constant() or left.right.is_constant())
    ):
        _fold_constant(tree)
    return False


def _same_group(*operators: int) -> bool:
    return set(operators) <= {11, 12} or set(operators) <= {13, 14}


def _fold_constant(tree: _Node) -> None:
    """Drop a constant child of the right child, which is a binary operator."""
    right = tree.right
    if right.right.is_constant():
        tree.right = right.left
    elif right.left.is_constant():
        additive = tree.value <= 12
        if tree.value == right.value:
            tree.value = 11 if additive else 13
        else:
            tree.value = 12 if additive else 14
        tree.right = right.right


def _postfix(tree: _Node) -> list[int]:
    """doe2vec ``generate_tree2exp`` followed by ``flatten``."""
    if tree.type() == 0:
        return [tree.value]
    if tree.type() == 1:
        return [*_postfix(tree.left), tree.value]
    return [*_postfix(tree.left), *_postfix(tree.right), tree.value]


_UNARY = {
    21: "-({})",
    22: "1/({})",
    23: "10*({})",
    24: "np.square({})",
    25: "np.sqrt(abs({}))",
    26: "abs({})",
    27: "np.round({})",
    28: "np.sin(2*np.pi*{})",
    29: "np.cos(2*np.pi*{})",
    30: "np.log(abs({}))",
    31: "np.exp({})",
    32: "np.sum({}, axis=1, keepdims=True)",
    33: "np.mean({}, axis=1, keepdims=True)",
    34: "np.cumsum({}, axis=1)",
    35: "np.prod({}, axis=1, keepdims=True)",
    36: "np.amax({}, axis=1, keepdims=True)",
}
_BINARY = {11: "+", 12: "-", 13: "*", 14: "/"}


def _expression(
    postfix: list[int], rng: random.Random, np_rng: np.random.RandomState, d: int
) -> str:
    """doe2vec ``generate_exp2fun`` without the noise operand 7, with shapes fixed.

    In doe2vec x1 and the reductions sum, mean, prod and max are ``(n,)`` and the
    translated vector works only at d = 2; here they are ``(n, 1)`` and ``(n, d)``.
    The rotated vector uses ``_dot``.
    """
    stack: list[str] = []
    for item in postfix:
        if item == 1:
            stack.append(str(rng.random() * 9 + 1))
        elif item == 2:
            stack.append("array_x")
        elif item == 3:
            stack.append("array_x[:,:1]")
        elif item == 4:
            stack.append("(np.hstack((array_x[:,1:], np.zeros((len(array_x), 1)))))")
        elif item == 5:
            stack.append(f"(dot(array_x, np.array({np_rng.rand(d, d).tolist()})))")
        elif item == 6:
            stack.append("(np.array(range(1, array_x.shape[1]+1)))")
        elif item in _BINARY:
            right = stack.pop()
            stack.append(stack.pop() + _BINARY[item] + right)
        elif item in _UNARY:
            stack.append(_UNARY[item].format(stack.pop()))
        else:
            raise ValueError(f"Operator {item} is not defined")
    return stack[0]


def _draw(rng: random.Random, np_rng: np.random.RandomState, dimension: int) -> str | None:
    """The expression of the next random tree, or None if the tree has noise.

    With fresh generators seeded with s, equal to doe2vec's ``generate_exp2fun(
    generate_tree2exp(generate_tree(6, 16)), n, dimension)`` after ``random.seed(s)``
    and ``np.random.seed(s)``. The noise operand (doe2vec's ``rand()``) is a fixed
    vector with one value per sample row, so not a function of x; such trees give None.
    """
    postfix = _postfix(_tree(rng, MIN_LENGTH, MAX_LENGTH))
    if 7 in postfix:
        return None
    return _expression(postfix, rng, np_rng, dimension)


@functools.cache
def _compile(expression: str) -> Any:
    return compile(expression, "<random tree>", "eval")


@dataclass(frozen=True)
class _RandomTree:
    """Objective given by a doe2vec expression of ``array_x``, ``(n, d) -> (n,)``."""

    expression: str

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Scalars are (n, 1); anything else fails here and the filter rejects it.
        return eval(_compile(self.expression), _NAMESPACE, {"array_x": x}).reshape(len(x))


def _accepted(function: _RandomTree, sample: np.ndarray) -> bool:
    """doe2vec's filter in ``generateData``, plus one value per point of ``sample``.

    Rejects a function whose values on ``sample`` are not finite, have any
    ``|y| < 1e-8`` or ``|y| > 1e8``, or have variance below 1.
    """
    try:
        with np.errstate(all="ignore"):
            y = np.asarray(function(sample), dtype=float)
    except (ValueError, IndexError):  # shape errors of doe2vec expressions
        return False
    return (
        y.shape == (len(sample),)
        and bool(np.all(np.isfinite(y)))
        and not np.any(np.abs(y) < 1e-8)
        and not np.any(np.abs(y) > 1e8)
        and np.var(y) >= 1.0
    )


def random_trees(dimensions: Iterable[int], instances: Iterable[int]) -> list[Problem]:
    """Random function trees (Tian et al. 2020) for the given dimensions and instances.

    A generator, not a fixed benchmark: every instance is a different function.
    Instance i draws trees from ``random.Random(i)`` and ``np.random.RandomState(i)``
    with doe2vec's generator (6 to 16 growth steps) until one passes doe2vec's filter
    on its sample, the first 2^8 points of the unscrambled Sobol sequence (which
    include the origin); see ``_accepted``. Trees with the noise operand are rejected
    too. The optimum is unknown. Ids are ``random_trees_f1_i{i}_d{d}``; the expression
    is ``problem.function.expression``. doe2vec's shape bugs are fixed, see
    docs/random-trees.md.
    """
    problems = []
    for instance in instances:
        for d in dimensions:
            sample = qmc.Sobol(d, scramble=False).random_base2(8)
            rng, np_rng = random.Random(instance), np.random.RandomState(instance)
            while True:
                expression = _draw(rng, np_rng, d)
                if expression is not None and _accepted(_RandomTree(expression), sample):
                    break
            problems.append(
                Problem(
                    id=ProblemId("random_trees", 1, instance, d),
                    function=_RandomTree(expression),
                    name="random_tree",
                )
            )
    return problems
