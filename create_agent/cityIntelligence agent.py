from dotenv import load_dotenv
load_dotenv()
from langchain.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain_tavily import TavilySearch
from rich import print
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
import os
import requests


API_KEY = os.getenv("OPENWEATHER_API_KEY")

#WEATHER TOOL
@tool
def get_weather(city: str) -> str:
    """
    Returns the current weather for a given city using OpenWeatherMap.
    """

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(url)

    if response.status_code != 200:
         return f"Error {response.status_code}: {response.text}"

    data = response.json()
    # print(data)

    weather = data["weather"][0]["description"]
    temperature = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]

    return (
        f"Weather in {city}:\n"
        f"Condition: {weather}\n"
        f"Temperature: {temperature}°C\n"
        f"Feels Like: {feels_like}°C\n"
        f"Humidity: {humidity}%"
    )



#NEWS AND INFO TOOL
tavily = TavilySearch(max_results=5)

@tool
def get_latest_news(city: str) -> str:
    """
    Returns the latest news headlines about the given city.
    """

    results = tavily.invoke(f"Latest news about {city}")

    if not results:
        return f"No recent news found for {city}."

    output = []

    for i, article in enumerate(results["results"], start=1):
        output.append(
            f"{i}. {article['title']}\n"
            f"{article['content']}\n"
            f"{article['url']}\n"
        )

    return "\n".join(output)

# Chatmodel
model = ChatMistralAI(model = "mistral-small-latest")


agent = create_agent(
    model= model,
    tools= [get_weather, get_latest_news],
    system_prompt="You are a helpful city assistant.",
)

#AGENT LOOP
print("\nCity Intelligence System | type 'exit' to Quit\n")
while True:
    query = input("\nYOU: ")
    if(query.lower() == 'exit'):
        break
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
    )
    print(result['messages'][-1].content)