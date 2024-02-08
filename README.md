# Interactive Markov Chain Weather Simulation

This project demonstrates an interactive Markov Chain model for simulating complex weather patterns using Python in a Jupyter Notebook environment. It utilizes `numpy` for numerical operations, `plotly.graph_objects` for visualization, and `ipywidgets` for interactive UI elements.

## Getting Started

### Prerequisites

Ensure you have the following packages installed in your Python environment:

- numpy
- plotly
- ipywidgets
- IPython

You can install these packages using pip:

```bash
pip install numpy plotly ipywidgets IPython
```

### Running the Simulation

1. **Open Jupyter Notebook**: Start by launching Jupyter Notebook in your project directory.

2. **Create a New Notebook**: In the Jupyter interface, create a new notebook.

3. **Copy the Python Code**: Copy the Python code provided in this repository into a cell in the notebook.

4. **Run the Cell**: Execute the cell by pressing `Ctrl + Enter`. This will display the interactive widgets, including sliders for adjusting the transition probabilities between different weather states and a button to update the model.

## How to Use

- **Adjust Transition Probabilities**: Use the sliders to adjust the probability of transitioning from one weather state to another. The states include Sunny, Cloudy, Rainy, Snowy, Windy, and Foggy.

- **Update Model**: After adjusting the probabilities, click the "Update Model" button to apply your changes. This will update the transition matrix and re-run the simulation with your specified parameters.

- **View Results**: The notebook will display two plots:
  - **Transition Matrix Heatmap**: Shows the probabilities of transitioning from one state to another.
  - **State Distribution Over Time**: Visualizes the probability of being in each state over a series of steps, based on the initial state and the transition matrix.

## Understanding the Model

- **Transition Matrix**: A square matrix used to describe the transitions between discrete states in a Markov chain. Each cell in the matrix represents the probability of moving from one state (row) to another state (column).

- **State Distribution Over Time**: Demonstrates how the probability distribution over the states evolves with each step in the Markov chain, starting from an initial state.

## Customization

You can customize the simulation by changing the number of steps, modifying the initial state, or adjusting the transition probabilities using the provided sliders.
