#creating a agent which will show the weather and latest news about a city.abs


from dotenv import load_dotenv
load_dotenv()
from langchain.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain_tavily import TavilySearch
from rich import print
from langchain_core.messages import HumanMessage

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


tools = {
    "get_latest_news":get_latest_news,
    "get_weather":get_weather
}

#CHAT HISTORY
messages = [

]

# chatmodel
model = ChatMistralAI(model = "mistral-small-latest")
model_with_tool = model.bind_tools([get_latest_news, get_weather])

print("\n\nWelcome to the CITY INTELLIGENCE AGENT. This agent is for fetching the weather and latest news of the given city. Type 'Exit' to Quit.\n")



#AGENT LOOP
while(True):
    query = input("\n\nYOU: ")
    messages.append(HumanMessage(query))
    if(query.lower() == 'exit'):
        break
    result = model_with_tool.invoke(messages)
    messages.append(result)

    if result.tool_calls:
        for tool_call in result.tool_calls:
            tool_name = tool_call['name']
            tool_result = tools[tool_name].invoke(tool_call)
            messages.append(tool_result)
            
        final_response = model_with_tool.invoke(messages)
        messages.append(final_response)

        print("\n\nAGENT: ", final_response.content, "\n\n")

    else:
        print("\n",result.content)

print("\nAGENT: Thanks for talking to me. See You soon!!!")

print(messages)
