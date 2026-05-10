from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langfuse.langchain import CallbackHandler

load_dotenv()


def main():
    print("Hello from langchanin-learning!")
    information = """
    Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, X, and xAI. Musk has been the wealthiest person in the world since 2025; as of May 2026, Forbes estimates his net worth to be US$788 billion.
    Born into the wealthy Musk family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.
    """

    summary_template = """
    given the information {information} about the person I want you to create:
    1. A short summary
    2. two interesting facts about them    
    """

    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)

    langfuse_handler = CallbackHandler()

    llm = ChatAnthropic(temperature=0, model="claude-haiku-4-5-20251001")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information}, config={"callbacks": [langfuse_handler]})
    print(response.content)

if __name__ == "__main__":
    main()
