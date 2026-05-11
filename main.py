from dotenv import load_dotenv

from typing import List
from pydantic import BaseModel, Field

from langgraph.prebuilt import create_react_agent

from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch

from langfuse import get_client
from langfuse.langchain import CallbackHandler

load_dotenv()

# @tool
# def get_weather(city: str) -> str:
#     """Get the current weather for a specific city.
#        Always use this tool when asked about weather.

#     Args:
#         city: The name of the city to get weather for.

#     Returns:
#         Current weather conditions for that city.
#     """
#     print(f"Getting weather for {city}")
#     return tavily.search(query=city)

class Source(BaseModel):
    """A single web source referenced by the agent when generating an answer."""

    url: str = Field(description="The full URL of the web page used as a source")

class AgentResponse(BaseModel):
    """Structured response returned by the agent, containing the answer and the sources it was derived from."""

    answer: str = Field(description="The agent's final answer to the user's query, synthesized from search results")
    sources: List[Source] = Field(description="List of web sources referenced to generate the answer", default_factory=list)

llm = ChatAnthropic(model="claude-opus-4-5")
tools = [TavilySearch(
    max_results=5,              # Max search results to return
    topic="general",            # Category of the search: "general", "news", or "finance"
    search_depth="advanced",    # Depth of the search: "basic", "advanced", "fast", or "ultra-fast"
    include_answer=True,        # Include a short answer to the original query in the results
    include_raw_content="markdown",  # Include cleaned HTML content of each result as markdown
    time_range=None,            # Filter by time: "day", "week", "month", "year", or None
    include_images=False,       # Include query-related images in the response
)]
agent = create_react_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-learning!")
    langfuse_handler = CallbackHandler()
    result = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather in NYC?")]},
        config={"callbacks": [langfuse_handler]},
    )
    print(result["messages"][-1].content)
    get_client().flush()


if __name__ == "__main__":
    main()