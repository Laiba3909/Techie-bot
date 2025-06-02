class OpenAIChatCompletionsModel:
    def __init__(self, model, openai_client):
        self.model = model
        self.openai_client = openai_client

    async def generate(self, prompt):
        response = await self.openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


class Agent:
    def __init__(self, name, instructions, model):
        self.name = name
        self.instructions = instructions
        self.model = model


    async def respond(self, user_input):
        prompt = f"{self.instructions}\nUser: {user_input}\nAgent:"
        return await self.model.generate(prompt)

class RunConfig:
    def __init__(self,model,tracing_disabled: bool, max_steps: int = 10, verbose: bool = False):
        self.max_steps = max_steps
        self.verbose = verbose
        self.model = model
        self.tracing_disabled = tracing_disabled

class Runner:
    def __init__(self, agent: Agent, run_config: RunConfig):
        self.agent = agent
        self.run_config = run_config
        self.step_count = 0

    @staticmethod
    async def run(agent: Agent, user_input: str, run_config: RunConfig):
        runner = Runner(agent, run_config)
        if runner.step_count >= run_config.max_steps:
            if run_config.verbose:
                print("Reached max steps limit. Stopping.")
            return None

        response = await agent.respond(user_input)

        if run_config.verbose:
            print(f"Step {runner.step_count + 1}: Query: {user_input}")
            print(f"Response: {response}\n")
        else:
            print(response)

        runner.step_count += 1

    
        class Result:
            def __init__(self, final_output):
                self.final_output = final_output

        return Result(response)