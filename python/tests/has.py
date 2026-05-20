import asyncio

from arcone_agent import Agent
from dotenv import load_dotenv
load_dotenv()

async def main() -> None:
    agent = Agent.from_env(
        system="Answer clearly and keep responses concise.",
        thinking=True,
        max_tokens=256,
    )

    text = await agent.ask_text("Explain arcone-agent in one paragraph.")
    print(text)


asyncio.run(main())