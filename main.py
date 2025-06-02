import os
from openai import AsyncOpenAI
from agent import OpenAIChatCompletionsModel
from agent import Agent
from agent import RunConfig
from agent import Runner
import chainlit as cl
from dotenv import load_dotenv


load_dotenv()

port = int(os.environ.get("PORT", 8000))
MODEL_NAME= 'gemini-2.0-flash'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = AsyncOpenAI(
      api_key = OPENAI_API_KEY,
      base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
        model=MODEL_NAME,
        openai_client=client
)

config = RunConfig(
        model=model,
        tracing_disabled=True
)



Developer_agent = Agent(
    name = 'Website Developer',
    instructions = 'You are a website developer you only give html css js code and also python, C++ Code if anyone needed and ask to you no other queries only code.',
    model = model
)

chef_agent = Agent(
    name = 'Recipie Agent',
    instructions = 'you are a chef assistant you help user to provide recipies only recipies they need nothing else no other queries only provide different recipies.',
    model = model
    
)

queries_agent = Agent(
    name = 'Queries Agent',
    instructions = 'You are a queiries agent you give all queries answer except recipies and html,css,js,python,c++ code you cant provide any code or recipes to user only another queries',
    model = model
)

manager = Agent(
    name = 'Manager Desicion maker',
    instructions=(
            "You are a manager agent. Based on the user input, classify the request into one of these categories:\n"
            "- 'website': if it's about creating, building, or designing a website.\n"
            "- 'recipe': if it's about cooking, food, or recipes.\n"
            "- 'other': for anything else.\n"
            "Reply with only one word: website, recipe, or other."
        ),
        model = model
)
conversation_log = []
@cl.on_message
async def main(message: cl.Message):
    user_input = message.content  
    
    conversation_log.append({"user_input": user_input, "agent_response": ""})

   
    history_text = ""
    for item in conversation_log:
        history_text += f"User: {item['user_input']}\n"
      
        if item['agent_response']:
            history_text += f"Agent: {item['agent_response']}\n"

   

    manager_decide = await Runner.run(manager, user_input, run_config=config)
    decision = manager_decide.final_output.lower().strip()
    
    if 'website' in decision:
        result = await Runner.run(Developer_agent , user_input, run_config=config)
    elif 'recipe' in decision:
        result = await Runner.run(chef_agent , user_input, run_config=config)
    elif 'other' in decision:
        result = await Runner.run(queries_agent , user_input, run_config=config)
    else:
        await cl.Message(content="Enter correct input, can't understand by manager").send()
        return
    
    await cl.Message(content=result.final_output).send()
    @cl.on_chat_start
    async def start():
       await cl.Message(content="Welcome! You can ask for code, recipes or general queries.").send()

