from langchain_community.tools import tool

@tool #decorator used to make the tool. A tool decorator when combined with a function creates a custom tool.
def get_greetings(name : str) -> str :  # tells that input will be a string and output will also be a string .
    """this tool is use to greet the user."""   #docstring or description. tells the llm what the tool is about. 
    return f"\n hello {name}, i hope you are doing well."

result2 = get_greetings.invoke({"name": "ritik"})

print(result2)

print("\n\n---------------------------------------------------------------------------------------\n\n")
print(get_greetings.description)
print(get_greetings.name)
print(get_greetings.args)   