import gradio as gr
import matplotlib.pyplot as plt
import numpy as np  # Import numpy
from PIL import Image
import random

# Import your existing modules/classes
from utils import read_cities_from_csv, create_distance_matrix, total_tour_distance, plot_tour
from branch_and_bound import TSPSolverBnB
from genetic_algorithm import TSPSolverGA
from dynamic_programming import TSPSolverHeldKarp


# Modified plot_tour_to_image function
def plot_tour_to_image(cities, tour=None, title="TSP Tour"):
    """Plots the TSP tour with modified line styles and returns it as a PIL Image object.
    If tour is None, it plots just the cities.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    x_coords = [city[1] for city in cities]
    y_coords = [city[2] for city in cities]

    ax.scatter(x_coords, y_coords, marker="o", color="blue")  # Plot cities as dots
    for city in cities:
        ax.text(city[1], city[2], city[0], fontsize=8, ha="right")

    if tour:  # Plot tour lines only if tour is provided
        tour_x_coords = [cities[i][1] for i in tour]
        tour_y_coords = [cities[i][2] for i in tour]
        tour_x_coords.append(cities[tour[0]][1])  # close the loop
        tour_y_coords.append(cities[tour[0]][2])

        total_distance = 0
        for i in range(len(tour)):
            start_city_index = tour[i]
            end_city_index = tour[(i + 1) % len(tour)]  # Wrap around to the start for the last segment
            start_city = cities[start_city_index]
            end_city = cities[end_city_index]

            distance = np.sqrt((end_city[1] - start_city[1]) ** 2 + (end_city[2] - start_city[2]) ** 2)
            total_distance += distance

            line_color = "red" if i != 0 else "green"  # First line green, others red
            line_style = "-"
            arrow_style = dict(arrowstyle="-|>", mutation_scale=10, lw=1)

            ax.arrow(
                start_city[1],
                start_city[2],
                end_city[1] - start_city[1],
                end_city[2] - start_city[2],
                color=line_color,
                linestyle=line_style,
                head_width=1,
                head_length=1,
                fc=line_color,
                ec=line_color,
            )

            mid_x = (start_city[1] + end_city[1]) / 2
            mid_y = (start_city[2] + end_city[2]) / 2
            ax.text(
                mid_x, mid_y + 1, f"{distance:.1f}", color="black", fontsize=8, ha="center", va="bottom"
            )  # Distance above line

        ax.set_title(f"{title} - Total Distance: {total_distance:.2f}")  # Add total distance to title
    else:
        ax.set_title(title)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return Image.fromarray(image_from_plot)


# New function to reset the plot
def reset_plot_ui():
    """Returns a blank image to reset the plot in the Gradio interface."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")  # Turn off axes for a blank plot
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return Image.fromarray(image_from_plot)


# --- Gradio Interface Logic ---


def generate_random_cities(num_cities):
    """Generates N random cities."""
    cities_data = []
    for i in range(int(num_cities)):
        city_name = f"City{i+1}"
        city_x = random.uniform(0, 100)  # Adjust range as needed
        city_y = random.uniform(0, 100)  # Adjust range as needed
        cities_data.append((city_name, city_x, city_y))
    return cities_data


def add_cities_click(num_cities):
    """Generates random cities and updates the plot."""
    cities_data = generate_random_cities(num_cities)
    initial_plot = plot_tour_to_image(cities_data, tour=None, title="Random Cities")  # Initial plot without tour
    return cities_data, initial_plot


def solve_tsp(algorithm_choice, city_data):
    """
    Solves the TSP based on the chosen algorithm and updates the plot.
    """
    if not city_data or len(city_data) < 2:
        return "Please add cities first or add at least two cities.", None

    cities = city_data  # city_data is already in the correct format

    dist_matrix = create_distance_matrix(cities)

    if algorithm_choice == "Branch and Bound":
        solver = TSPSolverBnB(dist_matrix)
    elif algorithm_choice == "Genetic Algorithm":
        solver = TSPSolverGA(
            dist_matrix, population_size=50, generations=200, mutation_rate=0.02
        )  # Use your GA parameters
    elif algorithm_choice == "Dynamic Programming":
        solver = TSPSolverHeldKarp(dist_matrix)
    else:
        return "Select a valid algorithm.", None

    best_tour, best_cost = solver.solve()
    if best_tour:
        tour_string = " -> ".join([cities[i][0] for i in best_tour] + [cities[best_tour[0]][0]])
        output_string = f"Best Tour: {tour_string}\nTotal Distance: {best_cost:.2f}"
        plot_image = plot_tour_to_image(cities, best_tour, title=f"TSP - {algorithm_choice}")
        return output_string, plot_image
    else:
        return "Could not find a tour.", None


# --- Gradio Interface Setup ---
with gr.Blocks(title="Traveling Salesperson Problem Solver") as demo:
    gr.Markdown("# Traveling Salesperson Problem Solver")
    gr.Markdown("Generate random cities, solve the TSP, and visualize the optimized route!")

    num_cities = gr.Number(label="Number of Cities", value=5, precision=0, minimum=2)  # Default 5 cities, minimum 2
    add_cities_button = gr.Button("Generate Cities")

    algorithm_choice = gr.Radio(
        ["Branch and Bound", "Genetic Algorithm", "Dynamic Programming"],  # Include Dynamic Programming
        label="Choose Algorithm",
        value="Branch and Bound",  # Default to BnB
    )

    solve_button = gr.Button("Solve TSP")

    reset_button = gr.Button("Reset Graph")  # Add Reset Button

    with gr.Row():
        with gr.Column():
            output_text = gr.Textbox(label="TSP Solution")
        with gr.Column():
            plot_output = gr.Image(label="Tour Plot")

    city_data_state = gr.State([])  # To store the generated cities data

    add_cities_button.click(
        add_cities_click, inputs=[num_cities], outputs=[city_data_state, plot_output]  # Update city_data_state and plot
    )

    solve_button.click(
        solve_tsp,
        inputs=[algorithm_choice, city_data_state],  # Use city_data_state as input
        outputs=[output_text, plot_output],
    )

    reset_button.click(reset_plot_ui, inputs=[], outputs=[plot_output])  # Add functionality for Reset Button


if __name__ == "__main__":
    demo.launch()
