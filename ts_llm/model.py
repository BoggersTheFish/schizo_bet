"""
Next-gen TS-LLM — graph concepts, dynamic edge tension, Schrödinger potentials,
multi-cluster attractors, damped wave dynamics, and diversity-aware generation.

**Wave recurrence (continuous step)** per edge::

    tension ← clip(tension * decay + sin(ω·time + φ) · amplitude)

**Learning (discrete, on input)**::

    tension ← tension + reinforcement    (bigram co-occurrence)
    amplitude ← amplitude + gain · reinforcement

``update_tension_wave(dt)`` advances ``time`` and applies the recurrence; input
calls ``input_sequence`` for activation, cross-cluster flow, collapse, and
reinforcement. Schrödinger nodes stay potential until activation + neighbor
tension exceed a threshold.

**Embeddings:** set ``embedding_mode`` to ``hash`` (deterministic, no deps),
``random``, ``dict`` (+ ``embedding_table`` / :meth:`from_embedding_json`), or
``sentence_transformer`` (requires ``pip install sentence-transformers``).
Similarity shapes edge reinforcement, cross-cluster flow, traversal weights,
and Schrödinger collapse toward cluster centroids.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ts_llm.embeddings import (
    DictEmbeddingBackend,
    EmbeddingBackend,
    HashEmbeddingBackend,
    build_backend,
    load_embedding_json,
    normalize_vector,
)


def _softmax(weights: Sequence[float], temperature: float) -> List[float]:
    if not weights:
        return []
    clean = [max(0.0, w) if math.isfinite(w) else 0.0 for w in weights]
    if not any(clean):
        return [1.0 / len(clean)] * len(clean)
    t = max(temperature, 1e-8)
    m = max(w / t for w in clean)
    exps = [math.exp((w / t) - m) for w in clean]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def _finite_nonneg(x: float, cap: float = 1e6) -> float:
    if not math.isfinite(x):
        return 0.0
    return max(0.0, min(x, cap))


@dataclass
class Node:
    """A concept or token: activation (flow) and internal tension (accumulated stress)."""

    node_id: str
    token: str
    schrodinger: bool = False
    collapsed: bool = False
    activation: float = 0.0
    internal_tension: float = 0.0
    """Aggregated incoming structural pressure; used for collapse and generation."""
    embedding_dim: int = 0
    embedding: List[float] = field(default_factory=list)

    def is_effective_active(self) -> bool:
        return (not self.schrodinger) or self.collapsed

    def collapse(self) -> None:
        self.collapsed = True


@dataclass
class Edge:
    """Directed u → v. Persistent ``tension``; wave uses ``wave_tick``, ``phase``, ``wave_amp``."""

    source: str
    target: str
    tension: float = 0.0
    """Learned structural weight + wave dynamics (clamped)."""
    wave_amp: float = 0.0
    """Oscillation amplitude; grows with reinforcement."""
    wave_tick: float = 0.0
    """Local phase time for sin(ω·time + φ)."""
    phase: float = 0.0


class TSLLM:
    """
    Graph-based TS language model: clusters, waves, Schrödinger collapse, diversity.

    **Core API:** ``input_sequence``, ``update_tension_wave``, ``get_clusters``,
    ``generate_output`` (supports ``repeat_*``, ``exploration_probability``,
    ``temperature``).

    **Dynamics:** multi-cluster activation flow, coherence-weighted generation,
    mid-generation wave steps and Schrödinger collapse for emergent sequences.

    **Embeddings:** optional backends influence learning and generation; see
    ``embedding_mode`` and :meth:`from_embedding_json`.
    """

    def __init__(
        self,
        collapse_threshold: float = 0.35,
        neighbor_tension_collapse_weight: float = 0.12,
        reinforcement: float = 0.25,
        propagation_strength: float = 0.4,
        propagation_steps: int = 3,
        damping: float = 0.02,
        contradiction_penalty: float = 0.15,
        wave_enabled: bool = True,
        wave_omega: float = 0.85,
        wave_tension_decay: Optional[float] = None,
        wave_amp_gain: float = 1.4,
        cross_cluster_strength: float = 0.22,
        cross_cluster_floor: float = 0.12,
        cluster_coherence_bonus: float = 0.35,
        multi_cluster_synthesis: float = 0.45,
        embedding_mode: str = "off",
        embedding_dim: int = 0,
        embedding_table: Optional[Dict[str, Sequence[float]]] = None,
        sentence_transformer_model: str = "all-MiniLM-L6-v2",
        embedding_cluster_affinity: float = 0.22,
        embedding_collapse_boost: float = 0.18,
        embedding_path_weight: float = 0.55,
        track_cluster_evolution: bool = False,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.collapse_threshold = collapse_threshold
        self.neighbor_tension_collapse_weight = neighbor_tension_collapse_weight
        self.reinforcement = reinforcement
        self.propagation_strength = propagation_strength
        self.propagation_steps = propagation_steps
        self.damping = damping
        self.contradiction_penalty = contradiction_penalty
        self.wave_enabled = wave_enabled
        self.wave_omega = wave_omega
        self.wave_tension_decay = (
            wave_tension_decay
            if wave_tension_decay is not None
            else damping * 0.25
        )
        self.wave_amp_gain = wave_amp_gain
        self.cross_cluster_strength = cross_cluster_strength
        self.cross_cluster_floor = cross_cluster_floor
        self.cluster_coherence_bonus = cluster_coherence_bonus
        self.multi_cluster_synthesis = multi_cluster_synthesis
        self.embedding_cluster_affinity = embedding_cluster_affinity
        self.embedding_collapse_boost = embedding_collapse_boost
        self.embedding_path_weight = embedding_path_weight
        self.track_cluster_evolution = track_cluster_evolution
        self.rng = rng or random.Random()

        mode = (embedding_mode or "off").lower().strip()
        if mode == "off" and embedding_dim > 0:
            mode = "random"
        dim_arg = embedding_dim
        self._embed_backend: Optional[EmbeddingBackend] = build_backend(
            mode,
            dim_arg,
            self.rng,
            sentence_transformer_model,
            embedding_table,
        )
        self.embedding_dim = (
            self._embed_backend.dimension if self._embed_backend is not None else 0
        )

        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[Tuple[str, str], Edge] = {}
        self._outgoing: Dict[str, List[str]] = defaultdict(list)
        self._incoming: Dict[str, List[str]] = defaultdict(list)
        self._contradictions: Set[Tuple[str, str]] = set()

        self._cluster_focus: Dict[int, float] = {}
        self._node_cluster_cache: Dict[str, int] = {}
        self._cluster_list_cache: List[Set[str]] = []
        self._cluster_evolution: List[Dict[str, Any]] = []

    @classmethod
    def from_embedding_json(cls, path: str, **kwargs: Any) -> TSLLM:
        """Build a model with ``embedding_mode='dict'`` from ``{\"token\": [float, ...]}`` JSON."""
        table = load_embedding_json(path)
        kwargs["embedding_mode"] = "dict"
        kwargs["embedding_table"] = table
        return cls(**kwargs)

    def _embed_for_token(self, text: str) -> List[float]:
        if self._embed_backend is None:
            return []
        return self._embed_backend.embed(text)

    def reembed_all_nodes(self) -> None:
        """Refresh every node's vector (e.g. after updating a dict table)."""
        if self._embed_backend is None:
            return
        for n in self._nodes.values():
            n.embedding = self._embed_for_token(n.token)
            n.embedding_dim = len(n.embedding)

    def merge_token_embeddings(self, table: Dict[str, Sequence[float]]) -> None:
        """Merge tokens into a dict backend and refresh node vectors."""
        if isinstance(self._embed_backend, DictEmbeddingBackend):
            self._embed_backend.update(table)
        elif self._embed_backend is None:
            d = len(next(iter(table.values())))
            self._embed_backend = DictEmbeddingBackend(
                dict(table), d, HashEmbeddingBackend(d)
            )
            self.embedding_dim = d
        else:
            raise TypeError(
                "merge_token_embeddings requires dict backend or no backend yet"
            )
        self.reembed_all_nodes()

    # --- graph construction ---

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def add_node(
        self,
        node_id: str,
        token: Optional[str] = None,
        schrodinger: bool = False,
    ) -> Node:
        if node_id in self._nodes:
            return self._nodes[node_id]
        tok = token if token is not None else node_id
        emb = self._embed_for_token(tok)
        edim = len(emb)
        n = Node(
            node_id=node_id,
            token=tok,
            schrodinger=schrodinger,
            collapsed=not schrodinger,
            embedding_dim=edim,
            embedding=emb,
        )
        self._nodes[node_id] = n
        return n

    def ensure_token(self, token: str, schrodinger: bool = False) -> str:
        self.add_node(token, token=token, schrodinger=schrodinger)
        return token

    def add_edge(self, source: str, target: str, initial_tension: float = 0.0) -> Edge:
        key = (source, target)
        if key in self._edges:
            e = self._edges[key]
            e.tension = max(e.tension, _finite_nonneg(initial_tension, 60.0))
            return e
        e = Edge(
            source=source,
            target=target,
            tension=_finite_nonneg(initial_tension, 60.0),
            phase=self.rng.uniform(0, 2 * math.pi),
        )
        self._edges[key] = e
        self._outgoing[source].append(target)
        self._incoming[target].append(source)
        return e

    def add_contradiction(self, a: str, b: str) -> None:
        self._contradictions.add(tuple(sorted((a, b))))

    def set_cluster_focus(self, weights: Dict[int, float]) -> None:
        """Higher-order attention: cluster index -> weight for cross-cluster flow and generation."""
        self._cluster_focus = dict(weights)

    def clear_cluster_focus(self) -> None:
        self._cluster_focus.clear()

    def _cluster_weight(self, cluster_index: int) -> float:
        if not self._cluster_focus:
            return 1.0
        return float(self._cluster_focus.get(cluster_index, 0.25))

    def _embedding_similarity(self, u: Node, v: Node) -> float:
        if self.embedding_dim <= 0 or not u.embedding or not v.embedding:
            return 1.0
        if len(u.embedding) != len(v.embedding):
            return 1.0
        dot = sum(a * b for a, b in zip(u.embedding, v.embedding))
        return 0.5 + 0.5 * dot

    def _path_embedding_blend(self, u: Optional[Node], v: Optional[Node]) -> float:
        if (
            self.embedding_dim <= 0
            or u is None
            or v is None
            or not u.embedding
            or not v.embedding
        ):
            return 1.0
        sim = self._embedding_similarity(u, v)
        w = self.embedding_path_weight
        return (1.0 - w) + w * sim

    def _max_centroid_alignment(self, n: Node, cluster_threshold: float) -> float:
        """How well the node aligns with any attractor centroid (cosine → [0,1])."""
        if not n.embedding:
            return 0.0
        d = len(n.embedding)
        clusters = self.get_clusters(cluster_threshold)
        best = -1.0
        for c in clusters:
            vecs = [
                self._nodes[x].embedding
                for x in c
                if x in self._nodes
                and self._nodes[x].embedding
                and len(self._nodes[x].embedding) == d
            ]
            if not vecs:
                continue
            mean = [
                sum(vecs[i][j] for i in range(len(vecs))) / len(vecs)
                for j in range(d)
            ]
            cent = normalize_vector(mean)
            dot = sum(a * b for a, b in zip(n.embedding, cent))
            best = max(best, dot)
        if best < -1.0:
            return 0.0
        return (best + 1.0) / 2.0

    def _clamp_edge(self, e: Edge) -> None:
        e.tension = _finite_nonneg(e.tension, 60.0)
        e.wave_amp = min(max(0.0, e.wave_amp), 8.0)

    def _touch_tension(self, src: str, tgt: str, delta: float) -> None:
        self.add_edge(src, tgt, 0.0)
        key = (src, tgt)
        e = self._edges[key]
        d = max(0.0, delta)
        e.tension = _finite_nonneg(e.tension + d, 60.0)
        e.wave_amp = min(8.0, e.wave_amp + self.wave_amp_gain * d)
        e.wave_tick = 0.0
        self._clamp_edge(e)

    # --- wave API ---

    def update_tension_wave(self, dt: float = 1.0) -> None:
        """
        Oscillatory convergence step::

            tension ← tension * (1 - wave_tension_decay) + sin(ω·time + φ) · amplitude
            amplitude ← amplitude · (1 - small_decay)

        Advance each edge's ``wave_tick`` by ``dt``. With ``wave_enabled=False``,
        only damping applies (no sinusoid).
        """
        if dt <= 0.0:
            return
        td = max(0.0, min(1.0, 1.0 - self.wave_tension_decay))
        amp_shrink = 1.0 - self.damping * 0.15
        for e in self._edges.values():
            e.wave_tick += dt
            osc = 0.0
            if self.wave_enabled:
                osc = math.sin(self.wave_omega * e.wave_tick + e.phase) * min(
                    e.wave_amp, 8.0
                )
            e.tension = _finite_nonneg(e.tension * td + osc, 60.0)
            e.wave_amp = max(0.0, e.wave_amp * amp_shrink)
            self._clamp_edge(e)

    # --- dynamics ---

    def clear_activations(self) -> None:
        for n in self._nodes.values():
            n.activation = 0.0

    def _decay_structural(self) -> None:
        factor = 1.0 - self.damping
        for e in self._edges.values():
            e.tension = _finite_nonneg(e.tension * factor, 60.0)
            e.wave_amp = max(0.0, e.wave_amp * (1.0 - self.damping * 0.35))
            self._clamp_edge(e)
        for n in self._nodes.values():
            n.internal_tension = _finite_nonneg(n.internal_tension * factor, 50.0)

    def _incoming_tension_sum(self, node_id: str) -> float:
        s = 0.0
        for src in self._incoming.get(node_id, []):
            key = (src, node_id)
            if key in self._edges:
                s += self._edges[key].tension
        return s

    def _update_internal_tensions(self) -> None:
        """Node tension state: blend of previous and neighbor edge tensions × activation."""
        beta = 0.55
        for nid, n in self._nodes.items():
            inc = self._incoming_tension_sum(nid)
            pressure = 0.0
            for src in self._incoming.get(nid, []):
                u = self._nodes.get(src)
                key = (src, nid)
                if u is None or key not in self._edges:
                    continue
                if u.is_effective_active():
                    pressure += self._edges[key].tension * (0.2 + 0.8 * u.activation)
            n.internal_tension = _finite_nonneg(
                (1.0 - beta) * n.internal_tension + beta * pressure,
                50.0,
            )

    def _propagate_activation(self) -> None:
        for _ in range(self.propagation_steps):
            next_act: Dict[str, float] = defaultdict(float)
            for (src, tgt), edge in self._edges.items():
                u, v = self._nodes.get(src), self._nodes.get(tgt)
                if u is None or v is None:
                    continue
                if not u.is_effective_active():
                    continue
                w = edge.tension * self._embedding_similarity(u, v)
                flow = self.propagation_strength * u.activation * w
                if v.is_effective_active() or v.schrodinger:
                    next_act[tgt] += flow
            for nid, add in next_act.items():
                n = self._nodes.get(nid)
                if n is None:
                    continue
                n.activation += add

    def _cross_cluster_activation_flow(self) -> None:
        """
        Multi-cluster interaction: push activation along cross-cluster edges so
        Schrödinger / distant attractors can entrain.
        """
        floor = self.cross_cluster_floor
        clusters = self.get_clusters(floor)
        self._cluster_list_cache = clusters
        node_cluster: Dict[str, int] = {}
        for i, c in enumerate(clusters):
            for nid in c:
                node_cluster[nid] = i
        self._node_cluster_cache = node_cluster

        extra: Dict[str, float] = defaultdict(float)
        for (u, v), e in self._edges.items():
            if e.tension < floor * 0.45:
                continue
            cu, cv = node_cluster.get(u), node_cluster.get(v)
            if cu is None or cv is None or cu == cv:
                continue
            nu = self._nodes.get(u)
            if nu is None or not nu.is_effective_active():
                continue
            wfocus = math.sqrt(
                self._cluster_weight(cu) * self._cluster_weight(cv)
            )
            nv = self._nodes.get(v)
            sem = self._embedding_similarity(nu, nv) if nv is not None else 1.0
            extra[v] += (
                self.cross_cluster_strength
                * e.tension
                * nu.activation
                * wfocus
                * sem
            )
        for nid, val in extra.items():
            n = self._nodes.get(nid)
            if n is not None:
                n.activation += val

    def _try_collapse(self, node_id: str) -> None:
        n = self._nodes.get(node_id)
        if n is None or not n.schrodinger or n.collapsed:
            return
        inc = self._incoming_tension_sum(node_id)
        score = n.activation + self.neighbor_tension_collapse_weight * math.log1p(
            inc
        )
        score += self.embedding_cluster_affinity * self._max_centroid_alignment(
            n, self.cross_cluster_floor
        )
        if score >= self.collapse_threshold:
            n.collapse()

    def _apply_contradictions(self, active_ids: Set[str]) -> None:
        for a, b in self._contradictions:
            if a in active_ids and b in active_ids:
                for x, y in ((a, b), (b, a)):
                    key = (x, y)
                    if key in self._edges:
                        e = self._edges[key]
                        e.tension = max(0.0, e.tension - self.contradiction_penalty)
                        self._clamp_edge(e)

    def _maybe_record_evolution(self, threshold: float) -> None:
        if not self.track_cluster_evolution:
            return
        clusters = self.get_clusters(threshold)
        self._cluster_evolution.append(
            {
                "threshold": threshold,
                "clusters": [sorted(c) for c in clusters],
                "mean_tension": (
                    sum(e.tension for e in self._edges.values())
                    / max(1, len(self._edges))
                ),
            }
        )

    def input_sequence(
        self,
        sequence: Sequence[str],
        *,
        schrodinger_mask: Optional[Sequence[bool]] = None,
        clear_activation_first: bool = True,
        evolution_threshold: Optional[float] = None,
    ) -> None:
        """
        Activate tokens, propagate (including cross-cluster), reinforce edges,
        collapse Schrödinger nodes using activation + neighbor tension.
        """
        if clear_activation_first:
            self.clear_activations()

        self._decay_structural()

        seq = list(sequence)
        if not seq:
            return

        active_ids: Set[str] = set()

        for i, tok in enumerate(seq):
            is_s = False
            if schrodinger_mask is not None and i < len(schrodinger_mask):
                is_s = bool(schrodinger_mask[i])
            nid = self.ensure_token(tok, schrodinger=is_s)
            n = self._nodes[nid]
            n.activation += 1.0
            active_ids.add(nid)

        self._propagate_activation()
        self._cross_cluster_activation_flow()
        self._update_internal_tensions()
        self._propagate_activation()

        for nid in list(self._nodes.keys()):
            self._try_collapse(nid)

        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            na, nb = self._nodes.get(a), self._nodes.get(b)
            if na is None or nb is None:
                continue
            if not na.is_effective_active() or not nb.is_effective_active():
                continue
            sim = self._embedding_similarity(na, nb)
            delta = self.reinforcement * (0.5 + 0.5 * na.activation) * (
                0.5 + 0.5 * nb.activation
            ) * sim
            self._touch_tension(a, b, delta)
            self._touch_tension(b, a, delta * 0.15)

        self._apply_contradictions(active_ids)

        eth = evolution_threshold if evolution_threshold is not None else self.cross_cluster_floor
        self._maybe_record_evolution(eth)

    # --- clusters ---

    def get_clusters(self, tension_threshold: float) -> List[Set[str]]:
        adj: Dict[str, Set[str]] = defaultdict(set)
        for (u, v), e in self._edges.items():
            if e.tension < tension_threshold:
                continue
            nu, nv = self._nodes.get(u), self._nodes.get(v)
            if nu is None or nv is None:
                continue
            if not nu.is_effective_active() or not nv.is_effective_active():
                continue
            adj[u].add(v)
            adj[v].add(u)

        seen: Set[str] = set()
        clusters: List[Set[str]] = []
        for start in adj:
            if start in seen:
                continue
            stack = [start]
            comp: Set[str] = set()
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                comp.add(x)
                for y in adj[x]:
                    if y not in seen:
                        stack.append(y)
            if comp:
                clusters.append(comp)
        return clusters

    def cluster_coherence(self, node_id: str, candidate: str, threshold: float) -> float:
        """Density of within-cluster support for moving to candidate (undirected)."""
        clusters = self.get_clusters(threshold)
        target_c = -1
        for i, c in enumerate(clusters):
            if candidate in c:
                target_c = i
                break
        if target_c < 0:
            return 0.0
        memb = clusters[target_c]
        if node_id not in memb:
            return 0.15
        internal = 0.0
        count = 0
        for u in memb:
            for v in memb:
                if u >= v:
                    continue
                k1, k2 = (u, v), (v, u)
                for k in (k1, k2):
                    if k in self._edges:
                        internal += self._edges[k].tension
                        count += 1
                        break
        return (internal / max(1, count)) / (1.0 + threshold)

    def multi_cluster_candidates(
        self,
        current: str,
        tension_threshold: float,
        top_k_clusters: int = 3,
    ) -> List[Tuple[str, float]]:
        clusters = self.get_clusters(tension_threshold)
        ranked = sorted(
            enumerate(clusters),
            key=lambda ic: sum(
                self._edges[(current, t)].tension
                for t in ic[1]
                if (current, t) in self._edges
            ),
            reverse=True,
        )
        weights: Dict[str, float] = defaultdict(float)
        for rank, (ci, memb) in enumerate(ranked[:top_k_clusters]):
            scale = 1.0 if rank == 0 else self.multi_cluster_synthesis / (rank + 1)
            scale *= self._cluster_weight(ci)
            for nid in memb:
                nu = self._nodes.get(nid)
                for tgt in self._outgoing.get(nid, []):
                    e = self._edges.get((nid, tgt))
                    if e:
                        nt = self._nodes.get(tgt)
                        blend = self._path_embedding_blend(nu, nt)
                        weights[tgt] += e.tension * scale * blend
        return sorted(weights.items(), key=lambda x: -x[1])

    def get_cluster_evolution(self) -> List[Dict[str, Any]]:
        return list(self._cluster_evolution)

    # --- generation ---

    def generate_output(
        self,
        start_node: Optional[str] = None,
        max_length: int = 20,
        temperature: float = 1.0,
        collapse_probability_on_visit: float = 0.45,
        creative_collapse_boost: float = 0.35,
        multi_cluster: bool = True,
        cluster_threshold: float = 0.18,
        repeat_token_window: int = 0,
        repeat_token_factor: float = 0.5,
        repeat_edge_memory: int = 0,
        repeat_edge_factor: float = 0.3,
        exploration_probability: float = 0.0,
        exploration: Optional[float] = None,
    ) -> List[str]:
        """
        Traverse active clusters: edge tension × coherence softmax at ``temperature``,
        with anti-loop diversity and optional uniform ``exploration_probability``.

        **Diversity:** ``repeat_token_window`` / ``repeat_token_factor`` downweight
        recent token strings; ``repeat_edge_memory`` / ``repeat_edge_factor`` penalize
        revisiting recent directed edges. ``exploration`` (deprecated alias) overrides
        ``exploration_probability`` when not ``None``.
        """
        if not self._nodes:
            return []

        p_explore = (
            exploration if exploration is not None else exploration_probability
        )

        if start_node is None:
            starters = [
                n.node_id
                for n in self._nodes.values()
                if n.is_effective_active() and self._outgoing.get(n.node_id)
            ]
            if not starters:
                starters = list(self._nodes.keys())
            current = self.rng.choice(starters)
        else:
            current = start_node
            self.add_node(current, token=current)

        out: List[str] = []
        path_tension = 0.0
        recent_edges: deque[Tuple[str, str]] = deque(maxlen=max(repeat_edge_memory, 1))

        for _ in range(max_length):
            n = self._nodes.get(current)
            if n is None:
                break

            if n.schrodinger and not n.collapsed:
                inc = self._incoming_tension_sum(current)
                align = self._max_centroid_alignment(n, cluster_threshold)
                p_collapse = collapse_probability_on_visit + creative_collapse_boost * (
                    1.0 - math.exp(-path_tension * 0.35)
                ) + 0.25 * math.tanh(self.neighbor_tension_collapse_weight * inc)
                p_collapse += self.embedding_collapse_boost * align
                p_collapse = min(0.92, max(0.05, p_collapse))
                if self.rng.random() < p_collapse:
                    n.collapse()
                if not n.collapsed:
                    nbrs = self._outgoing.get(current, [])
                    if not nbrs:
                        break
                    if p_explore > 0.0 and self.rng.random() < p_explore:
                        current = self.rng.choice(nbrs)
                    else:
                        ws_s = []
                        for t in nbrs:
                            nt = self._nodes.get(t)
                            te = (
                                _finite_nonneg(self._edges[(current, t)].tension)
                                if (current, t) in self._edges
                                else 0.01
                            )
                            ws_s.append(te * self._path_embedding_blend(n, nt))
                        if sum(ws_s) <= 1e-12:
                            current = self.rng.choice(nbrs)
                        else:
                            pr = _softmax(ws_s, temperature)
                            current = self.rng.choices(nbrs, weights=pr, k=1)[0]
                    continue

            out.append(n.token)

            nbrs = self._outgoing.get(current, [])
            if not nbrs:
                break

            if multi_cluster:
                cand = self.multi_cluster_candidates(current, cluster_threshold)
                pool = [c for c, _ in cand if c in nbrs]
                if len(pool) >= 2:
                    nbrs = pool
                ws = []
                for t in nbrs:
                    e = self._edges[(current, t)]
                    coh = _finite_nonneg(self.cluster_coherence(current, t, cluster_threshold), 10.0)
                    nt = self._nodes.get(t)
                    nt_boost = 1.0 + 0.2 * min(nt.internal_tension if nt else 0.0, 25.0)
                    pb = self._path_embedding_blend(n, nt)
                    base = _finite_nonneg(e.tension) * nt_boost * (
                        1.0 + self.cluster_coherence_bonus * coh
                    ) * pb
                    ws.append(_finite_nonneg(base))
            else:
                ws = []
                for t in nbrs:
                    e = self._edges[(current, t)]
                    nt = self._nodes.get(t)
                    pb = self._path_embedding_blend(n, nt)
                    ws.append(_finite_nonneg(e.tension * pb))

            if repeat_token_window > 0 and repeat_token_factor > 0.0:
                recent = out[-repeat_token_window :]
                for i, t in enumerate(nbrs):
                    ntok = self._nodes[t].token
                    c = sum(1 for x in recent if x == ntok)
                    if c:
                        ws[i] *= repeat_token_factor**c

            if repeat_edge_memory > 0 and repeat_edge_factor > 0.0:
                frozen = frozenset(recent_edges) if len(recent_edges) else frozenset()
                for i, t in enumerate(nbrs):
                    if (current, t) in frozen:
                        ws[i] *= repeat_edge_factor

            if sum(ws) <= 1e-12:
                ws = [1.0] * len(nbrs)
            probs = _softmax(ws, temperature)
            if not all(math.isfinite(p) for p in probs):
                probs = [1.0 / len(nbrs)] * len(nbrs)
            if p_explore > 0.0 and self.rng.random() < p_explore:
                nxt = self.rng.choice(nbrs)
            else:
                nxt = self.rng.choices(nbrs, weights=probs, k=1)[0]
            path_tension += self._edges[(current, nxt)].tension
            if repeat_edge_memory > 0:
                recent_edges.append((current, nxt))
            current = nxt

            nn = self._nodes.get(current)
            if nn and nn.schrodinger and not nn.collapsed:
                inc = self._incoming_tension_sum(current)
                al = self._max_centroid_alignment(nn, cluster_threshold)
                p2 = (
                    creative_collapse_boost
                    + 0.15 * math.tanh(path_tension)
                    + 0.1 * math.tanh(inc)
                    + self.embedding_collapse_boost * al
                )
                if self.rng.random() < min(0.88, p2):
                    nn.collapse()

            if self.wave_enabled and self.rng.random() < 0.25:
                self.update_tension_wave(self.rng.uniform(0.15, 0.45))

        return out

    def train_corpus(
        self,
        sequences: Iterable[Sequence[str]],
        epochs: int = 1,
    ) -> None:
        for _ in range(epochs):
            for seq in sequences:
                self.input_sequence(seq)

    def snapshot_tensions(self) -> Dict[Tuple[str, str], float]:
        return {k: e.tension for k, e in self._edges.items()}
