from __future__ import annotations

import time

from arcone_agent import Document, InMemoryKnowledgeBase, InMemorySessionStore, runtime_info


def bench(label: str, count: int, fn) -> None:
    started = time.perf_counter()
    for _ in range(count):
        fn()
    elapsed = time.perf_counter() - started
    print(f"{label}: {elapsed:.6f}s total, {(elapsed / count) * 1_000_000:.2f}us/op")


def main() -> None:
    print(runtime_info())
    bench("InMemorySessionStore()", 10_000, InMemorySessionStore)
    bench(
        "Document.text(...)",
        10_000,
        lambda: Document.text(
            "doc",
            "Arcone exposes Rust agent capabilities through Python.",
            metadata={"source": "bench"},
        ),
    )
    bench("InMemoryKnowledgeBase()", 10_000, InMemoryKnowledgeBase)


if __name__ == "__main__":
    main()
