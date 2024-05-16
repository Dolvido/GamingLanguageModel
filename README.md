# Snake Game with Language Model Agent

This project demonstrates the integration of a language model agent with a classic Snake game environment. The agent interacts with the game environment using natural language commands and receives feedback on the game state.

## Installation

1. Clone the repository:
```
git clone https://github.com/your-username/snake-game-language-model.git
```

2. Install the required dependencies:
```
pip install gymnasium numpy langchain_community ollama
```

## Usage

1. Run the `main.py` script:
```
python main.py
```
2. The agent will start interacting with the Snake game environment. It will provide commands to control the snake's movement based on the current game state.

3. The game state will be displayed in the console, showing the snake's head as 'h', body segments as 's', and the food as 'f'.

4. The agent will continue playing the game until the snake collides with the walls or its own body.

## How it Works

- The Snake game environment is implemented using the Gymnasium library.
- The language model agent is powered by the Ollama model from the `langchain_community` library.
- The agent receives the current game state as an observation and generates actions based on the state.
- The actions are communicated to the game environment using a JSON format, specifying the action type and input.
- The game environment updates the state based on the agent's actions and provides feedback to the agent.
- The agent continues to interact with the game environment until the game is over.

## Game Environment

The Snake game environment is implemented in the `SnakeEnv` class. It includes the following key components:

- `step(action)`: Updates the game state based on the given action and returns the new observation, reward, done flag, and info.
- `reset()`: Resets the game environment to its initial state.
- `render()`: Renders the current game state in the console.

## Language Model Agent

The language model agent is implemented using the Ollama model and the `initialize_agent` function from the `langchain` library. The agent is configured with the following:

- `snake_tool`: A custom tool that interacts with the Snake game environment. It takes the action as input and returns the updated game state.
- `zero-shot-react-description`: The agent type that allows the agent to generate actions based on the game state without prior training.

The agent follows a specific format for interacting with the game environment:

- It receives the current game state as an observation.
- It generates thoughts and actions based on the observation.
- It communicates the actions to the game environment using a JSON blob.
- It receives the updated game state as an observation and continues the interaction loop.

## Future Enhancements

- Implement a graphical user interface for the Snake game.
- Experiment with different language models and agent configurations.
- Add more complex game mechanics and obstacles.
- Train the agent using reinforcement learning techniques.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).