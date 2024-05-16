import gymnasium as gym
import numpy as np
from langchain_community.llms import Ollama
from langchain.agents import load_tools, initialize_agent, Tool
import json

class SnakeEnv(gym.Env):
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)  # 0: up, 1: right, 2: down, 3: left
        self.snake_pos = None
        self.food_pos = None
        self.direction = None
        self.done = False

    def step(self, action):
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        new_head = (self.snake_pos[0][0] + dx, self.snake_pos[0][1] + dy)
        
        if self._is_collision(new_head):
            self.done = True
            reward = -1
        elif new_head == self.food_pos:
            self.snake_pos.insert(0, new_head)
            self._place_food()
            reward = 1
        else:
            self.snake_pos.pop()
            self.snake_pos.insert(0, new_head)
            reward = 0
        
        observation = self._get_observation()
        return observation, reward, self.done, {}

    def reset(self):
        self.snake_pos = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = self.action_space.sample()
        self._place_food()
        self.done = False
        observation = self._get_observation()
        return observation

    def render(self, mode='human'):
        grid = np.full((self.grid_size, self.grid_size), ' ')
        grid[self.snake_pos[0]] = 'h'
        for cell in self.snake_pos[1:]:
            grid[cell] = 's'
        grid[self.food_pos] = 'f'
        print(grid)

    def _is_collision(self, head):
        return (
            head[0] < 0 or head[0] >= self.grid_size or
            head[1] < 0 or head[1] >= self.grid_size or
            head in self.snake_pos
        )

    def _place_food(self):
        while True:
            food_pos = (
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size)
            )
            if food_pos not in self.snake_pos:
                self.food_pos = food_pos
                break

    def _get_observation(self):
        observation = np.full((self.grid_size, self.grid_size), 0)
        observation[self.snake_pos[0]] = 1
        for cell in self.snake_pos[1:]:
            observation[cell] = 2
        observation[self.food_pos] = 3
        return observation

def snake_tool(action_str):
    try:
        action_json = json.loads(action_str)
        action_input = action_json["action_input"].lower()
        action_mapping = {
            "north": 0,
            "east": 1,
            "south": 2,
            "west": 3
        }
        action = action_mapping.get(action_input)
        
        if action is None:
            return "Invalid action. Please enter a valid cardinal direction (North, South, East, West)."
        
        observation, reward, done, info = env.step(action)
        if done:
            env.reset()
        
        # Convert observation to a grid representation
        grid = np.full((env.grid_size, env.grid_size), ' ')
        grid[env.snake_pos[0]] = 'h'
        for cell in env.snake_pos[1:]:
            grid[cell] = 's'
        grid[env.food_pos] = 'f'
        
        return  f"""Current observation:
                {grid}

                You are controlling a snake in a grid environment. The snake's goal is to eat the food (represented by 'f') while avoiding collisions with the walls or its own body.

                The snake's head is represented by 'h', and the body segments are represented by 's'.

                You have access to these tools

                [snake]


                The way you use the tools is by specifying a JSON blob. Specifically, this JSON should have an "action" key (with the name of the tool to use) and an "action_input" key (with the input to the tool going here). The only value that should be in the "action" field is: "Snake". 
                The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:
                ```
                {
                "action": "Snake",
                "action_input": "North"
                }
                ```
                ALWAYS use the following format:
                Question: the input question you must answer
                Thought: you should always think about what to do
                Action:
                ```
                $JSON_BLOB
                ```
                Observation: the result of the action
                ... (this Thought/Action/Observation can repeat N times)
                Thought: I now know the final answer
                Final Answer: the final answer to the original input question
                Remember, the "action_input" should be a cardinal direction (North, South, East, or West) representing the direction the snake should move. Do not include any other text or information in the "action_input" field.
                """
    except (json.JSONDecodeError, KeyError):
        return "Invalid JSON format. Please provide a valid JSON blob with 'action' and 'action_input' keys."


snake_tool = Tool(
    name="Snake",
    func=snake_tool,
    description="Interact with the Snake environment. The input should be a cardinal direction (North, South, East, West) representing the action.",
)

llm = Ollama(model="mistral")
tools = [snake_tool]
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True, handle_parsing_errors=True)

env = SnakeEnv(grid_size=5)
observation = env.reset()

while True:
    action = agent.run(f"Current observation: {observation}. What action should be taken?")
    action = action.strip().lower()

    if action not in ["north", "south", "east", "west"]:
        print("Invalid action. Please provide a valid cardinal direction (North, South, East, West).")
        continue

    observation, reward, done, info = env.step(env.action_space.sample())
    env.render()
    if done:
        observation = env.reset()