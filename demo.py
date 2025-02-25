import gradio as gr
import math
import matplotlib.pyplot as plt
from io import StringIO  # Import StringIO
import numpy as np  # Import numpy
from PIL import Image
import csv
# Import your existing modules/classes
from utils import read_cities_from_csv, create_distance_matrix, total_tour_distance, plot_tour
from branch_and_bound import TSPSolverBnB
from genetic_algorithm import TSPSolverGA  # Import the GA solver (assuming you have it)


# Helper function to convert plot to image (for Gradio compatibility)
def plot_tour_to_image(cities, tour, title="TSP Tour"):
    """Plots the TSP tour and returns it as a PIL Image object."""
    x_coords = [cities[i][1] for i in tour]
    y_coords = [cities[i][2] for i in tour]
    x_coords.append(cities[tour[0]][1])
    y_coords.append(cities[tour[0]][2])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x_coords, y_coords, marker='o', linestyle='-')
    for idx in tour:
        ax.text(cities[idx][1], cities[idx][2], cities[idx][0], fontsize=8, ha='right')
    ax.set_title(title)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return Image.fromarray(image_from_plot)


# --- Gradio Interface Logic ---

def add_cities_ui(num_cities):
    """
    Dynamically creates input fields for the specified number of cities.
    """
    city_inputs = []
    for i in range(int(num_cities)):  # Ensure num_cities is an integer
        with gr.Row():  # Put each city's inputs on a separate row
                city_name = gr.Textbox(label=f"City {i+1} Name", value=f"City{i+1}")
                city_x = gr.Number(label=f"City {i+1} X", value=0, precision=1)
                city_y = gr.Number(label=f"City {i+1} Y", value=0, precision=1)
                city_inputs.extend([city_name, city_x, city_y])
    return city_inputs


def solve_tsp(algorithm_choice, city_data):
    """
    Solves the TSP based on the chosen algorithm.
    """
    # Process city data
    cities = []
    for i in range(0, len(city_data), 3):
      if i + 2 < len(city_data):  # Ensure we don't go out of bounds
        city_name = city_data[i]
        city_x = city_data[i + 1]
        city_y = city_data[i + 2]
        if city_name and city_x is not None and city_y is not None:  # Check all parts exist
            cities.append((city_name.strip(), float(city_x), float(city_y)))

    if len(cities) < 2:
          return "Add at least two cities to calculate a tour.", None  # Changed return

    dist_matrix = create_distance_matrix(cities)

    if algorithm_choice == "Branch and Bound":
        solver = TSPSolverBnB(dist_matrix)
    elif algorithm_choice == "Genetic Algorithm":
        solver = TSPSolverGA(dist_matrix, population_size=50, generations=200, mutation_rate=0.02)  # Use your GA parameters
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
    gr.Markdown("Define the number of cities, add their coordinates, and find the shortest route!")

    num_cities = gr.Number(label="Number of Cities", value=2, precision=0) # Default 2 city
    add_cities_button = gr.Button("Add Cities")

    city_inputs = gr.State([])  # Store dynamically created inputs
    with gr.Column(visible=False) as city_input_column:
        dynamic_city_inputs = gr.List() # Use gr.List()


    algorithm_choice = gr.Radio(
        ["Branch and Bound", "Genetic Algorithm", "Algorithm 2 (Placeholder)"],
        label="Choose Algorithm",
        value="Branch and Bound"  # Default to BnB
    )

    solve_button = gr.Button("Solve TSP")

    with gr.Row():
        with gr.Column():
            output_text = gr.Textbox(label="TSP Solution")
        with gr.Column():
            plot_output = gr.Image(label="Tour Plot")


    def add_cities_click(num_cities):
      new_city_inputs = add_cities_ui(num_cities)
      return {city_input_column: gr.Column(visible=True), dynamic_city_inputs: new_city_inputs, city_inputs: new_city_inputs} #update the city_input


    add_cities_button.click(add_cities_click, [num_cities], [city_input_column, dynamic_city_inputs, city_inputs])
    solve_button.click(solve_tsp, [algorithm_choice, city_inputs], [output_text, plot_output])


if __name__ == "__main__":
    demo.launch()