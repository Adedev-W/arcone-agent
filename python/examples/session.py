import asyncio

from arcone_agent import Agent, InMemorySessionStore


async def main() -> None:
    store = InMemorySessionStore()
    first = Agent.from_env(
        session_id="python-session-demo",
        session_store=store,
        thinking=False,
        max_tokens=128,
    )

    print(await first.ask_text("My project codename is Arcone. Remember it."))

    second = Agent.from_env(
        session_id="python-session-demo",
        session_store=store,
        thinking=False,
        max_tokens=128,
    )
    print(await second.ask_text("What project codename did I mention?"))


if __name__ == "__main__":
    asyncio.run(main())
