# Interactive Markov Chain Weather Simulation

This project demonstrates an interactive Markov Chain model for simulating complex weather patterns using Python in a Jupyter Notebook environment. It utilizes `numpy` for numerical operations, `plotly.graph_objects` for visualization, `ipywidgets` for interactive UI elements and `SciPy` for advanced mathematical computations, particularly in solving systems of linear equations which are essential for calculating the steady state distribution and first passage times in the Markov Chain model.

## Getting Started

### Prerequisites

Ensure you have the following packages installed in your Python environment:

- numpy
- plotly
- ipywidgets
- IPython
- scipy

You can install these packages using pip:

```bash
pip install numpy plotly ipywidgets IPython scipy
```

### Running the Simulation

1. **Open Jupyter Notebook**: Start by launching Jupyter Notebook in your project directory.

2. **Create a New Notebook**: In the Jupyter interface, create a new notebook.

3. **Copy the Python Code**: Copy the Python code provided in this repository into a cell in the notebook.

4. **Run the Cell**: Execute the cell by pressing `Ctrl + Enter`. This will display the interactive widgets, including sliders for adjusting the transition probabilities between different weather states and a button to update the model.

## How to Use

- **Adjust Transition Probabilities**: Use the sliders to adjust the probability of transitioning from one weather state to another. The states include Sunny, Cloudy, Rainy, Snowy, Windy, and Foggy.

- **Set Simulation Steps**: Use the slider to set the number of steps you want the simulation to run.

- **Auto-update Model**: Check the "Auto-update" box to automatically apply changes and update the model as you adjust the sliders or the initial state. Unch6eck this box if you prefer to manually update the model using the "Update Model" button.

- **Update Model**: After adjusting the probabilities, click the "Update Model" button to apply your changes. This will update the transition matrix and re-run the simulation with your specified parameters.

- **Reset**: Click the "Reset" button to revert all settings to their initial values. This is useful for starting a new simulation from scratch.

- **View Results**: The notebook will display four plots:
  - **Probability Over Time**: Graphs the probability of being in the target state over time, providing insights into the dynamics of transitioning into the selected state.
  - **Transition Matrix Heatmap**: Shows the probabilities of transitioning from one state to another.
  - **State Distribution Over Time**: Visualizes the probability of being in each state over a series of steps, based on the initial state and the transition matrix.
  - **Steady State Distribution**: A bar chart that represents the long-term or steady state distribution of the weather states. This distribution indicates the probability of being in each state after a large number of steps, providing insight into the long-term behavior of the weather system under the current transition probabilities.
  - **First Passage Times**: A heatmap showing the expected number of steps required to first transition from one state to another. This visualization offers an understanding of how quickly or slowly the system might transition into each state for the first time from a given starting state.

## Understanding the Model

- **Probability Over Time**: This feature allows users to visualize the probability of transitioning to a specific target state over time.

- **Convergence Proof**: The `convergence_proof_with_markov_chains` function assesses whether the Markov chain will reach a steady state by checking for matrix regularity.

- **Transition Matrix**: A square matrix used to describe the transitions between discrete states in a Markov chain. Each cell in the matrix represents the probability of moving from one state (row) to another state (column).

- **State Distribution Over Time**: Demonstrates how the probability distribution over the states evolves with each step in the Markov chain, starting from an initial state.

- **Steady State Distribution**: This concept refers to the equilibrium condition of the Markov chain, where the probabilities of being in each state remain constant over time despite continuous transitions. The steady state distribution is calculated from the transition matrix and represents the long-term behavior of the system. It's an important metric for understanding which states are most dominant or stable as the number of transitions approaches infinity.

- **First Passage Times**: First passage time is the expected number of steps required for the system to enter a given state for the first time, starting from another state. This metric provides insights into the temporal dynamics of state transitions, indicating how quickly or slowly the system is likely to experience new states. The matrix of first passage times can help identify the most and least accessible states within the system.


## Customization

The simulation offers several customization options, including the ability to change the number of steps, modify the initial state, and adjust the transition probabilities using the provided sliders and dropdown menu.

![image](https://github.com/Flubbeh/WeatherTransitionMarkovModel/assets/26907138/348d4b2c-c715-4197-acbb-22cbc90a87c4)
![image](https://github.com/Flubbeh/WeatherTransitionMarkovModel/assets/26907138/bcb69597-696b-4897-9246-09484ea60550)
![image](https://github.com/Flubbeh/WeatherTransitionMarkovModel/assets/26907138/063c5c9e-baaf-43d4-b8b4-427478e83288)
![image](https://github.com/Flubbeh/WeatherTransitionMarkovModel/assets/26907138/94b317ed-6b4e-4578-ab73-01a91822d70f)
![image](https://github.com/Flubbeh/WeatherTransitionMarkovModel/assets/26907138/6a8af43f-f04d-4cb4-b23e-3b6dacbdf62f)
![image](https://github.com/Flubbeh/WeatherTransitionMarkovModel/assets/26907138/f9d758a3-70d5-4104-a3a9-c80dcd99b3d2)
![image](https://github.com/Flubbeh/WeatherTransitionMarkovModel/assets/26907138/e1d2e514-1395-4d42-9482-22f84821ee9f)

