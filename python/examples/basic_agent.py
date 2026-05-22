import asyncio

from arcone_agent import Agent


async def main() -> None:
    agent = Agent.from_env(
        system="Answer clearly and keep responses concise.",
        thinking=True,
        max_tokens=256,
    )

    response = await agent.ask("Explain arcone-agent in one paragraph.")
    print(response.content)
    print(f"finish_reason={response.finish_reason}")


if __name__ == "__main__":
    asyncio.run(main())
