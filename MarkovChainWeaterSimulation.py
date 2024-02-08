import numpy as np
import plotly.graph_objects as go
from scipy.linalg import solve
import plotly.io as pio
from numpy.linalg import matrix_power
from IPython.display import display as ipydisplay
from ipywidgets import interactive, HBox, VBox, widgets, Layout, Button, Output, Dropdown, Checkbox, HTML
from IPython.display import display
pio.renderers.default = 'notebook+jupyterlab+plotly_mimetype+notebook_connected'


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


def calculate_steady_state(matrix):
    dim = matrix.shape[0]
    # Ensure matrix is square
    if dim != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    # Adjust matrix to add steady state condition
    q = matrix[:-1]  # Take all rows except the last to keep matrix square after adding the row of ones
    q = np.vstack((q, np.ones(dim)))  # Add a row of ones for the steady state condition

    # Create the right-hand side of the equations
    b = np.zeros(dim)
    b[-1] = 1  # Set the last element to 1 for the steady state condition

    # Solve for the steady state distribution
    pi = solve(q, b)
    return pi


def first_passage_times(matrix):
    dim = matrix.shape[0]
    I = np.eye(dim)
    F = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            if i != j:
                Q = np.copy(matrix)
                Q[:, j] = 0  # Remove transitions to absorbing state j
                N = np.linalg.inv(I - Q)  # Fundamental matrix
                F[i, j] = N[i, :].sum()
    return F


def plot_steady_state_distribution(pi, states):
    fig = go.Figure(data=go.Bar(x=states, y=pi))
    fig.update_layout(
        title="Steady State Distribution",
        xaxis=dict(title="State"),
        yaxis=dict(title="Probability")
    )

    steady_state_explanation = HTML(
        value="""
    <p><b>Steady State Distribution:</b> This distribution represents the long-term behavior of the Markov chain, 
    showing the probability of being in each state after a large number of steps. It's calculated by solving 
    a system of linear equations derived from the transition matrix, ensuring that the sum of probabilities equals 1.</p>
    """
    )
    ipydisplay(steady_state_explanation)
    ipydisplay(fig)


def plot_first_passage_times(fpt_matrix, states):
    fig = go.Figure(data=go.Heatmap(z=fpt_matrix, x=states, y=states, colorscale="Cividis"))
    fig.update_layout(
        title="First Passage Times",
        xaxis=dict(title="To State"),
        yaxis=dict(title="From State")
    )

    first_passage_times_explanation = HTML(
        value="""
        <p><b>First Passage Times:</b> This concept refers to the expected number of steps needed to reach a particular state 
        from another state for the first time. The matrix of first passage times provides insights into the dynamics of 
        transitions between states, showing how quickly or slowly the system might move into each state.</p>
        """
    )
    ipydisplay(first_passage_times_explanation)
    ipydisplay(fig)


def interactive_complex_markov_chain(initial_steps=10):
    intro_text = HTML(
        value="""
        <h4>Welcome to the Interactive Complex Markov Chain Model!</h4>
        <p>This tool helps you understand how Markov chains work, specifically in modeling weather transitions. 
        A Markov chain is a mathematical system that undergoes transitions from one state to another, 
        with the probability of each state dependent only on the current state and not on the history of states.</p>
        """
    )
    matrix = create_complex_weather_transition_matrix()
    states = ["Sunny", "Cloudy", "Rainy", "Snowy", "Windy", "Foggy"]
    initial_state_dropdown = Dropdown(options=states, value='Sunny', description='Initial State:')
    steps_slider = widgets.IntSlider(value=initial_steps, min=1, max=50, step=1, description='Steps:',
                                     style={'description_width': 'initial'})
    auto_update_checkbox = Checkbox(value=False, description='Auto-update')
    output = Output()

    sliders = [
        [widgets.FloatSlider(value=matrix[i, j], min=0, max=1.0, step=0.01, description=f'{states[i]} → {states[j]}:',
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

            # Plot the transition matrix
            plot_transition_matrix(new_matrix, states)

            # Simulate Markov chain and plot the state distribution over time
            simulate_markov_chain(new_matrix, steps, states, initial_state)

            # Calculate and plot the steady state distribution
            pi = calculate_steady_state(new_matrix)
            plot_steady_state_distribution(pi, states)

            # Calculate and plot the first passage times
            fpt_matrix = first_passage_times(new_matrix)
            plot_first_passage_times(fpt_matrix, states)

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
    ui_components = [intro_text, HBox(slider_boxes), initial_state_dropdown, steps_slider,
                     auto_update_checkbox, update_button, output]
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
    matrix_explanation = HTML(
        value="""
        <p><b>Transition Matrix:</b> This matrix represents the probabilities of moving from one weather state 
        to another. Each row sums up to 1, indicating the total probability of transitioning to any state 
        from the current state.</p>
        """
    )
    ipydisplay(matrix_explanation)
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
    state_distribution_explanation = HTML(
        value="""
        <p><b>State Distribution Over Time:</b> This visualization shows how the probability of each state evolves over the specified number of steps. 
        It helps in understanding the dynamics of the Markov chain, illustrating how the system transitions from the initial state to other states over time. 
        This is crucial for observing how quickly the system reaches a steady state or how it responds to changes in the transition matrix.</p>
    """
    )
    ipydisplay(state_distribution_explanation)
    fig.show()


interactive_complex_markov_chain(initial_steps=10)