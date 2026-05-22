import asyncio

from arcone_agent import Agent


async def lookup_price(args: dict) -> dict:
    return {
        "symbol": args["symbol"].upper(),
        "price": 128.40,
        "currency": "USD",
    }


async def main() -> None:
    agent = Agent.from_env(
        system="Use tools when they help answer market questions.",
        thinking=True,
        max_tokens=256,
    )
    agent.add_tool(
        name="lookup_price",
        description="Return a demo market quote for a ticker symbol.",
        schema={
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
        handler=lookup_price,
    )

    print(await agent.ask_text("What is the demo quote for ACME?"))


if __name__ == "__main__":
    asyncio.run(main())
