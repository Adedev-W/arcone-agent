import asyncio

from arcone_agent import Agent, StreamingUnsupportedError


async def main() -> None:
    agent = Agent.from_env(
        system="Answer clearly and keep responses concise.",
        thinking=False,
        max_tokens=256,
    )

    stream = await agent.stream("Write a short status update for arcone-agent.")
    try:
        async for delta in stream:
            print(delta, end="", flush=True)
    except StreamingUnsupportedError as exc:
        print(f"\nstreaming stopped: {exc}")
        return

    response = await stream.finish()
    print(f"\nfinish_reason={response.finish_reason}")


if __name__ == "__main__":
    asyncio.run(main())
