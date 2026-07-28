from dotenv import load_dotenv
load_dotenv()

from langchain_tavily import TavilySearch
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

tool = TavilySearch(
    max_results=5,
)

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Summarize the following news into clear bullet points"),
    ("human", "{news}")
])

model = ChatMistralAI(model="mistral-small-latest")
parser = StrOutputParser()

pipeline = summary_prompt | model | parser
# Although TavilySearch is a Runnable, it is not directly added to this pipeline
# because its output (search results) does not match the input format expected
# by ChatPromptTemplate, which requires {"news": ...}. A mapping Runnable
# (e.g., RunnableLambda or RunnablePassthrough.assign()) would be needed
# to connect them in a single pipeline.


query = input("News_summarizer : Tell about the news you want to summarize: \nYou: ")
news_result = tool.run(query)
# We can use both run() and invoke() to execute the TavilySearch tool.
# run() is the older convenience method, while invoke() is the newer,
# standard Runnable method recommended in modern LangChain.


result = pipeline.invoke({"news": news_result})

print("\n\n")
print(f"News_summarizer : {result}")

print("\n\n---------------------------------------------------------------------------------------\n\n")
print(tool.description)
print(tool.name)
print(tool.args)