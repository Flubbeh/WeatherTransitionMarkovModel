import numpy as np
import plotly.graph_objects as go
from ipywidgets import interactive, HBox, VBox, widgets, Layout, Button, Output
from IPython.display import display


def create_complex_weather_transition_matrix():
    """
    Creates a transition matrix for a complex weather model.
    """
    # Format: [Sunny, Cloudy, Rainy, Snowy, Windy, Foggy]
    matrix = np.array([
        [0.5, 0.2, 0.1, 0.05, 0.1, 0.05],  # Sunny
        [0.2, 0.3, 0.25, 0.05, 0.1, 0.1],  # Cloudy
        [0.1, 0.3, 0.4, 0.1, 0.05, 0.05],  # Rainy
        [0.05, 0.1, 0.1, 0.6, 0.1, 0.05],  # Snowy
        [0.1, 0.2, 0.1, 0.1, 0.4, 0.1],  # Windy
        [0.05, 0.15, 0.1, 0.05, 0.15, 0.5]  # Foggy
    ])
    return matrix


def interactive_complex_markov_chain(steps=10):
    matrix = create_complex_weather_transition_matrix()
    states = ["Sunny", "Cloudy", "Rainy", "Snowy", "Windy", "Foggy"]

    # Create sliders for each transition probability
    sliders = [[widgets.FloatSlider(
        value=matrix[i, j],
        min=0,
        max=1.0,
        step=0.01,
        description=f'{states[i]}->{states[j]}:',
        readout_format='.2f',
        style={'description_width': 'initial'}
    ) for j in range(len(states))] for i in range(len(states))]

    output = Output()  # For dynamic updates

    def update_matrix(button):
        with output:
            output.clear_output()  # Clear the previous plots
            new_matrix = np.array([[slider.value for slider in row] for row in sliders])
            # Normalize the matrix
            new_matrix = new_matrix / new_matrix.sum(axis=1, keepdims=True)
            plot_transition_matrix(new_matrix, states)
            simulate_markov_chain(new_matrix, steps, states, initial_state=0)

    update_button = Button(description="Update Model")
    update_button.on_click(update_matrix)

    slider_boxes = [VBox([s for s in row]) for row in sliders]
    ui = VBox([HBox(slider_boxes), update_button, output])

    display(ui)
    with output:
        plot_transition_matrix(matrix, states)  # Display the initial matrix
        simulate_markov_chain(matrix, steps, states, initial_state=0)  # Simulate with the initial matrix


def plot_transition_matrix(matrix, states):
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=states,
        y=states,
        colorscale="Viridis",
    ))
    fig.update_layout(
        title="Transition Matrix",
        xaxis=dict(title="To State"),
        yaxis=dict(title="From State"),
    )
    fig.show()


def simulate_markov_chain(matrix, steps, states, initial_state=0):
    state_history = np.zeros((steps, len(matrix)))
    state_history[0, initial_state] = 1  # Initialize the starting state

    for i in range(1, steps):
        state_history[i] = np.dot(state_history[i - 1], matrix)  # Transition to the next state

    fig = go.Figure()
    for state_index, state in enumerate(states):
        fig.add_trace(go.Scatter(
            x=list(range(steps)),
            y=state_history[:, state_index],
            mode='lines+markers',
            name=state
        ))

    fig.update_layout(
        title="State Distribution Over Time",
        xaxis_title="Step",
        yaxis_title="Probability",
        legend_title="Weather States"
    )
    fig.show()


interactive_complex_markov_chain(steps=10)