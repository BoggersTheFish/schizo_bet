"""Run: python3 -m ts_llm"""

import json

from ts_llm.model import TSLLM


def main() -> None:
    m = TSLLM(
        collapse_threshold=0.2,
        reinforcement=0.3,
        damping=0.01,
        wave_enabled=True,
        wave_omega=0.9,
        cross_cluster_strength=0.25,
        track_cluster_evolution=True,
        embedding_mode="hash",
        embedding_dim=96,
        rng=None,
    )
    corpus = [
        "the cat sat on the mat".split(),
        "the dog sat on the log".split(),
        "the cat sat still".split(),
    ]
    m.train_corpus(corpus, epochs=2)
    for _ in range(4):
        m.update_tension_wave(0.4)

    print("Clusters (threshold=0.15):")
    for i, c in enumerate(m.get_clusters(0.15)):
        print(f"  {i}: {sorted(c)}")

    m.set_cluster_focus({0: 1.0, 1: 0.6})
    gen_kw = dict(
        max_length=16,
        temperature=1.12,
        repeat_token_window=10,
        repeat_token_factor=0.46,
        repeat_edge_memory=8,
        repeat_edge_factor=0.2,
        exploration_probability=0.14,
    )
    print("\nGenerated:", " ".join(m.generate_output(**gen_kw)))
    m.clear_cluster_focus()

    ev = m.get_cluster_evolution()
    print("\nCluster evolution steps:", len(ev))
    if ev:
        print("Last snapshot:", json.dumps(ev[-1], indent=2)[:400], "...")

    m2 = TSLLM(
        collapse_threshold=0.22,
        neighbor_tension_collapse_weight=0.15,
        embedding_mode="hash",
        embedding_dim=64,
        rng=None,
    )
    m2.add_node("ghost", schrodinger=True)
    for _ in range(5):
        m2.input_sequence(["idea", "ghost", "form"], clear_activation_first=True)
    g = m2.get_node("ghost")
    print("\nSchrödinger 'ghost' collapsed:", g.collapsed if g else None)
    print(
        "Sample:",
        " ".join(
            m2.generate_output(
                start_node="idea",
                max_length=12,
                multi_cluster=True,
                creative_collapse_boost=0.4,
                temperature=1.15,
                repeat_token_window=12,
                repeat_token_factor=0.32,
                repeat_edge_memory=8,
                repeat_edge_factor=0.12,
                exploration_probability=0.3,
            )
        ),
    )


if __name__ == "__main__":
    main()
