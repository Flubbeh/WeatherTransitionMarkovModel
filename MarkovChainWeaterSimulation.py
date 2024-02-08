import numpy as np
import plotly.graph_objects as go
from ipywidgets import interactive, HBox, VBox, widgets, Layout, Button, Output, Dropdown, Checkbox
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


def interactive_complex_markov_chain(initial_steps=10):
    matrix = create_complex_weather_transition_matrix()
    states = ["Sunny", "Cloudy", "Rainy", "Snowy", "Windy", "Foggy"]
    initial_state_dropdown = Dropdown(options=states, value='Sunny', description='Initial State:')
    steps_slider = widgets.IntSlider(value=initial_steps, min=1, max=50, step=1, description='Steps:',
                                     style={'description_width': 'initial'})
    auto_update_checkbox = Checkbox(value=False, description='Auto-update')
    output = Output()

    sliders = [
        [widgets.FloatSlider(value=matrix[i, j], min=0, max=1.0, step=0.01, description=f'{states[i]}->{states[j]}:',
                             readout_format='.2f', style={'description_width': 'initial'})
         for j in range(len(states))] for i in range(len(states))
    ]

    def validate_matrix(matrix):
        # Normalize the matrix rows to sum to 1
        return np.nan_to_num(matrix / matrix.sum(axis=1, keepdims=True), nan=0.0)

    def add_observers():
        for row in sliders:
            for slider in row:
                slider.observe(update_model, names='value')
        steps_slider.observe(update_model, names='value')
        initial_state_dropdown.observe(update_model, names='value')

    def remove_observers():
        for row in sliders:
            for slider in row:
                slider.unobserve(update_model, names='value')
        steps_slider.unobserve(update_model, names='value')
        initial_state_dropdown.unobserve(update_model, names='value')

    def update_model(change=None):
        with output:
            output.clear_output(wait=True)  # Clear the previous plots
            new_matrix = np.array([[slider.value for slider in row] for row in sliders])
            new_matrix = validate_matrix(new_matrix)  # Normalize the matrix
            initial_state = states.index(initial_state_dropdown.value)
            steps = steps_slider.value
            plot_transition_matrix(new_matrix, states)
            simulate_markov_chain(new_matrix, steps, states, initial_state)

    def auto_update_change(change):
        if change.new:
            add_observers()
            update_model()
        else:
            remove_observers()

    auto_update_checkbox.observe(auto_update_change, names='value')

    update_button = Button(description="Update Model")
    update_button.on_click(update_model)

    slider_boxes = [VBox([s for s in row]) for row in sliders]
    ui_components = [HBox(slider_boxes), initial_state_dropdown, steps_slider, auto_update_checkbox, update_button,
                     output]
    ui = VBox(ui_components)
    display(ui)
    update_model()  # Initial call to display the model and simulation


def plot_transition_matrix(matrix, states):
    fig = go.Figure(data=go.Heatmap(z=matrix, x=states, y=states, colorscale="Viridis"))
    fig.update_layout(
        title="Transition Matrix",
        xaxis=dict(title="To State"),
        yaxis=dict(title="From State")
    )
    fig.show()


def simulate_markov_chain(matrix, steps, states, initial_state=0):
    state_history = np.zeros((steps, len(states)))
    state_history[0, initial_state] = 1  # Initialize the starting state
    for i in range(1, steps):
        state_history[i] = np.dot(state_history[i - 1], matrix)  # Transition to the next state
    fig = go.Figure()
    for state_index, state in enumerate(states):
        fig.add_trace(
            go.Scatter(x=list(range(steps)), y=state_history[:, state_index], mode='lines+markers', name=state))
    fig.update_layout(
        title="State Distribution Over Time",
        xaxis_title="Step",
        yaxis_title="Probability",
        legend_title="Weather States"
    )
    fig.show()


interactive_complex_markov_chain(initial_steps=10)