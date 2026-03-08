from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MODEL = "qwen3:1.7b"
MAX_ITERATIONS = 10
# llm = init_chat_model(model=MODEL)
# -- Tools (LangChain @tool decorator) --


@tool
def get_product_price(productName: str) -> float:
    """Get the price of a product."""
    products = {"laptop": 999.99, "phone": 199.99, "tablet": 299.99}
    print(f">> Executing get_product_price(product='{productName}')")
    return products.get(productName, 0.0)


@tool
def apple_discount(productPrice: float, discountTier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(
        f">> Executing apple_discount(productPrice=' {productPrice} ',discountTier=' {discountTier} ')"
    )
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discountTier, 0)
    return round(productPrice * (1 - discount / 100), 2)


@traceable(name="Langchain Agent Loop")
def run_agent_loop(query: str):
    tools = [get_product_price, apple_discount]
    tool_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(model=f"ollama:{MODEL}", temperature=0.0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"Question: {query}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content="You are a helpful shopping assistant. "
            "You have access to a product catalog tool "
            "and a discount tool. \n\n"
            "STRICT RULES - you must follow these exactly: \n"
            "1. NEVER guess or assume any product price. "
            "You MUST call get_product_price first to get the real price. \n"
            "2. Only call apple_discount AFTER you have received "
            "a price from get_product_price. Pass the exact price "
            "returned by get_product_price - do NOT pass a made-up number. \n"
            "3. NEVER calculate discounts yourself using math. "
            "Always use the apple_discount tool. \n"
            "4. If the user does not specify a discount tier, ask them which tier to use - do NOT assume one."
        ),
        HumanMessage(content=query),
    ]
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print(f"\nThe Final Answer is {ai_message.content}")
            return ai_message.content
        tool_name = tool_calls[0].get("name")
        tool_args = tool_calls[0].get("args", {})
        tool_call_id = tool_calls[0].get("id")
        tool = tool_dict[tool_name]
        print(f"[Tool Selected] {tool_name} with args: {tool_args}")
        tool_to_use = tool_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        observation = tool_to_use.invoke(tool_args)
        print(f" [Tool Result] {observation}")
        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )
    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (bind_tools)!")
    query = "What is the price of the laptop  after applying the gold discounts?"
    result = run_agent_loop(query)
