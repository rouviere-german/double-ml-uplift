"""Directed acyclic graphs, d-separation and backdoor reachability.

The d-separation test is the Bayes-Ball style reachability algorithm of Koller and Friedman,
*Probabilistic Graphical Models* (2009), Algorithm 3.1: first collect the ancestors of the
conditioning set, then explore trails from the source, tracking whether each node is entered from
a parent ("down") or from a child ("up"). A trail is blocked at a non-collider in the
conditioning set and at a collider none of whose descendants is in the conditioning set.

It is implemented here rather than taken from a library so that the repository's own tests can
check it against brute-force enumeration of every simple path
(``tests/unit/test_graph.py::test_d_separation_matches_brute_force_path_enumeration``) and
against a hypothesis-generated family of random DAGs.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

__all__ = ["CausalGraph", "CycleError", "GraphError", "UnknownNodeError"]

_Direction = Literal["up", "down"]


class GraphError(ValueError):
    """The graph or a query is malformed."""


class CycleError(GraphError):
    """The declared edges contain a directed cycle, so the graph is not a DAG."""


class UnknownNodeError(GraphError):
    """A query mentioned a node that is not in the graph."""


@dataclass(frozen=True)
class CausalGraph:
    """An immutable DAG over string-named nodes.

    ``parents`` and ``children`` are materialised once; every query is read-only, so a graph can
    be shared between requests without copying.
    """

    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    _parents: Mapping[str, frozenset[str]]
    _children: Mapping[str, frozenset[str]]

    @classmethod
    def from_edges(cls, nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> CausalGraph:
        """Build a graph, rejecting unknown endpoints, self-loops and directed cycles."""
        node_tuple = tuple(dict.fromkeys(nodes))
        if not node_tuple:
            raise GraphError("a causal graph needs at least one node")
        known = set(node_tuple)
        parents: dict[str, set[str]] = {name: set() for name in node_tuple}
        children: dict[str, set[str]] = {name: set() for name in node_tuple}
        edge_list: list[tuple[str, str]] = []
        for source, target in edges:
            if source not in known or target not in known:
                raise UnknownNodeError(f"edge {source!r} -> {target!r} mentions an unknown node")
            if source == target:
                raise CycleError(f"self-loop on {source!r}")
            if source in parents[target]:
                continue  # duplicate edge
            parents[target].add(source)
            children[source].add(target)
            edge_list.append((source, target))
        graph = cls(
            nodes=node_tuple,
            edges=tuple(edge_list),
            _parents={name: frozenset(values) for name, values in parents.items()},
            _children={name: frozenset(values) for name, values in children.items()},
        )
        graph._require_acyclic()
        return graph

    # --- structure -----------------------------------------------------------------------

    def _require_acyclic(self) -> None:
        indegree = {name: len(self._parents[name]) for name in self.nodes}
        queue = deque(name for name, degree in indegree.items() if degree == 0)
        seen = 0
        while queue:
            name = queue.popleft()
            seen += 1
            for child in sorted(self._children[name]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if seen != len(self.nodes):
            remaining = sorted(name for name, degree in indegree.items() if degree > 0)
            raise CycleError(f"the declared edges contain a cycle among {remaining}")

    def check_nodes(self, names: Iterable[str]) -> None:
        unknown = sorted(set(names) - set(self.nodes))
        if unknown:
            raise UnknownNodeError(f"unknown node(s): {', '.join(unknown)}")

    def parents(self, node: str) -> frozenset[str]:
        self.check_nodes([node])
        return self._parents[node]

    def children(self, node: str) -> frozenset[str]:
        self.check_nodes([node])
        return self._children[node]

    def ancestors(self, nodes: Iterable[str], *, inclusive: bool = True) -> frozenset[str]:
        """All nodes with a directed path into ``nodes`` (the nodes themselves if inclusive)."""
        seeds = list(nodes)
        self.check_nodes(seeds)
        seen: set[str] = set()
        stack = list(seeds)
        while stack:
            name = stack.pop()
            if name in seen:
                continue
            seen.add(name)
            stack.extend(self._parents[name])
        if not inclusive:
            seen -= set(seeds)
        return frozenset(seen)

    def descendants(self, nodes: Iterable[str], *, inclusive: bool = True) -> frozenset[str]:
        """All nodes reachable from ``nodes`` by directed edges."""
        seeds = list(nodes)
        self.check_nodes(seeds)
        seen: set[str] = set()
        stack = list(seeds)
        while stack:
            name = stack.pop()
            if name in seen:
                continue
            seen.add(name)
            stack.extend(self._children[name])
        if not inclusive:
            seen -= set(seeds)
        return frozenset(seen)

    def without_outgoing_edges(self, node: str) -> CausalGraph:
        """The mutilated graph ``G_underline_T`` used to test the backdoor criterion."""
        self.check_nodes([node])
        return CausalGraph.from_edges(self.nodes, (edge for edge in self.edges if edge[0] != node))

    def undirected_neighbours(self, node: str) -> frozenset[str]:
        return self._parents[node] | self._children[node]

    # --- d-separation --------------------------------------------------------------------

    def reachable(self, sources: Iterable[str], conditioned: Iterable[str]) -> frozenset[str]:
        """Nodes d-connected to ``sources`` given ``conditioned``.

        Koller and Friedman, *Probabilistic Graphical Models* (2009), Algorithm 3.1.
        """
        source_set = set(sources)
        condition_set = set(conditioned)
        self.check_nodes(source_set | condition_set)
        ancestors_of_conditioned = self.ancestors(condition_set) if condition_set else frozenset()

        visited: set[tuple[str, _Direction]] = set()
        reached: set[str] = set()
        stack: list[tuple[str, _Direction]] = [(name, "up") for name in source_set]
        while stack:
            node, direction = stack.pop()
            if (node, direction) in visited:
                continue
            visited.add((node, direction))
            if node not in condition_set:
                reached.add(node)
            if direction == "up" and node not in condition_set:
                stack.extend((parent, "up") for parent in self._parents[node])
                stack.extend((child, "down") for child in self._children[node])
            elif direction == "down":
                if node not in condition_set:
                    stack.extend((child, "down") for child in self._children[node])
                if node in ancestors_of_conditioned:
                    stack.extend((parent, "up") for parent in self._parents[node])
        return frozenset(reached)

    def is_d_separated(
        self, first: Iterable[str], second: Iterable[str], conditioned: Iterable[str] = ()
    ) -> bool:
        """True when no active trail connects ``first`` and ``second`` given ``conditioned``."""
        first_set, second_set = set(first), set(second)
        condition_set = set(conditioned)
        self.check_nodes(first_set | second_set | condition_set)
        overlap = (first_set | second_set) & condition_set
        if overlap:
            raise GraphError(f"cannot condition on a queried node: {', '.join(sorted(overlap))}")
        if first_set & second_set:
            raise GraphError("the two node sets overlap")
        return not (self.reachable(first_set, condition_set) & second_set)

    # --- brute-force oracle (used by tests and by refusal messages) -----------------------

    def simple_paths(self, source: str, target: str, *, limit: int = 50_000) -> Iterator[list[str]]:
        """Every simple undirected path between two nodes, shortest first.

        ``limit`` bounds the number of partial trails explored, so a dense graph cannot make a
        request run forever. Enumeration is exponential in the worst case and is used for error
        messages and as a test oracle; :meth:`is_d_separated` is the linear-time path.
        """
        self.check_nodes([source, target])
        queue: deque[list[str]] = deque([[source]])
        explored = 0
        while queue:
            path = queue.popleft()
            explored += 1
            if explored > limit:
                return
            node = path[-1]
            if node == target:
                yield path
                continue
            for neighbour in sorted(self.undirected_neighbours(node)):
                if neighbour not in path:
                    queue.append([*path, neighbour])

    def is_collider(self, path: Sequence[str], position: int) -> bool:
        before, node, after = path[position - 1], path[position], path[position + 1]
        return before in self._parents[node] and after in self._parents[node]

    def path_is_blocked(self, path: Sequence[str], conditioned: Iterable[str]) -> bool:
        """Block a trail at a conditioned non-collider, or at an unconditioned collider."""
        condition_set = set(conditioned)
        for position in range(1, len(path) - 1):
            node = path[position]
            if self.is_collider(path, position):
                if not (self.descendants([node]) & condition_set):
                    return True
            elif node in condition_set:
                return True
        return False

    def open_paths(
        self, source: str, target: str, conditioned: Iterable[str] = ()
    ) -> list[list[str]]:
        """Trails from ``source`` to ``target`` left open by ``conditioned``, shortest first."""
        condition_set = set(conditioned)
        return [
            path
            for path in self.simple_paths(source, target)
            if not self.path_is_blocked(path, condition_set)
        ]

    def backdoor_paths(self, treatment: str, outcome: str) -> list[list[str]]:
        """Trails from treatment to outcome whose first edge points into the treatment."""
        parents = self._parents[treatment]
        return [
            path
            for path in self.simple_paths(treatment, outcome)
            if len(path) > 1 and path[1] in parents
        ]
