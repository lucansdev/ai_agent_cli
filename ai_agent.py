from langchain_openai import ChatOpenAI
from langchain.agents import tool,AgentExecutor,create_tool_calling_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import subprocess
import os
import dotenv

dotenv.load_dotenv()
llm = ChatOpenAI(openai_api_key=os.getenv("openaiKey"),model="gpt-4.1-mini")

prompt = PromptTemplate.from_template("""
Analyze the customer's query and the conversation history. If the customer requests something related to CLI commands, assist them using the `comandos_cli` tool. If the question does not involve shell commands, simply respond directly based on the conversation history, if relevant.
Do not execute the same command multiple times; if there is already an output, show that output to the user.

Conversation History:
{history}

Customer Query: {input}

{agent_scratchpad}
""")


memory = ConversationBufferMemory(memory_key="history", return_messages=True) 

@tool
def comandos_cli(text: str):
    """executa comandos no shell do computador"""
    resposta = subprocess.run(text, shell=True, capture_output=True, check=True)
    return resposta.stdout.decode("utf-8")

tools = [comandos_cli]

agent = create_tool_calling_agent(llm=llm,tools=tools,prompt=prompt)

agent_executor = AgentExecutor(memory=memory,agent=agent,tools=tools,verbose=True)

while True:

    query = input("qual seu comandos: ")
    try:
        result = agent_executor.invoke({"input":query})
        print(result["output"])
    except Exception as e:
        print(f"Error during agent run: {e}")