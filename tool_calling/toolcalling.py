# creating a tool
# tool binding
# tool calling
# tool execution

from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage
from rich import print

# 1.) creating a tool

@tool
def get_text_length(text : str) -> int :
    """count the number of character in the given text and return it"""
    return len(text)

tools = {
    "get_text_length" : get_text_length
}

# 2.)tool binding
model = ChatMistralAI(model = "mistral-small-latest")
model_with_tools = model.bind_tools([get_text_length])

messages = [

]

prompt = input("\nYOU: ")


messages.append(HumanMessage(prompt))

result = model_with_tools.invoke(messages)
messages.append(result)


# 3.) Tool calling
if result.tool_calls:
    tool_calls = result.tool_calls[0]
    tool_name = tool_calls['name']
    tool_args = tool_calls['args']


# print(tool_calls)
# print(tool_name)
# print(tool_args)


# 4.) tool execution 
# tool_result = tool_name.invoke(tool_args)
    tool_message = tools[tool_name].invoke(tool_calls)      # first of all, llm jo tool suggest kr rha h mai uss tool ko invoke krunga. Agar mai uss tool ko args k saath invoke krta hu toh mujhe sirf tool ka jo kaam h uss kaam ka result milega but agar mai usse tool ko args ki jagah pr AI msg ke tool calls k saath invoke krta hu toh mujhe "Tool Message" milega. 
    messages.append(tool_message)
   

# 5.) sending the response back to the llm
    final_response = model_with_tools.invoke(messages)
    messages.append(final_response)
    print(final_response.content)
    print("\n")
    # print(messages)

else:
    print(result.content)