import asyncio

from arcone_agent import (
    Agent,
    Document,
    InMemoryKnowledgeBase,
    InMemoryVectorRetriever,
    KnowledgeAgent,
    OpenAiEmbedder,
)


async def main() -> None:
    knowledge = InMemoryKnowledgeBase(max_chars=1200, overlap_chars=120)
    chunks = await knowledge.add_document(
        Document.text(
            "arcone-overview",
            "Arcone combines agents, tools, sessions, retrieval, and team routing.",
            title="Arcone Overview",
            source="example",
        )
    )

    retriever = InMemoryVectorRetriever(OpenAiEmbedder.from_env())
    await retriever.index(chunks)

    base_agent = Agent.from_env(
        system="Answer only from the retrieved context.",
        thinking=True,
        max_tokens=256,
    )
    agent = KnowledgeAgent.from_agent(base_agent, retriever, top_k=4)

    response = await agent.ask("What does Arcone combine?")
    print(response.content)
    for source in response.sources:
        print(f"[{source.index}] {source.title} score={source.score:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
