from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

model = ChatMistralAI(model="mistral-small-latest")
parser = StrOutputParser()

code_Prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Code Generator. Please do not explain the code or give any examples. Just generate it."),
    ("human", "{Query}")
])

explain_Prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who explains code in simple terms. Explain the code line by line(Pointwise). Also dont include to many hashtags, emojis and stars, just keep it sleek and simple."),
    ("human", "Explain the following code in simple words: \n {Code}")
])


code = code_Prompt | model | parser 




parallel = RunnableParallel({
    "passthrough" : RunnablePassthrough(), 
    "explanation" : explain_Prompt | model | parser
}
)

pipeline = code | parallel

question = input("Write the name of the code and explanation you want: \n")
result = pipeline.invoke(question)

print("\n\n")
print(result["passthrough"])
print("\n\n\n\n")
print(result["explanation"])