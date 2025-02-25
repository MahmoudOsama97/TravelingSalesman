import math
import csv
import random
import matplotlib.pyplot as plt

def read_cities_from_csv(filepath):
    """
    Reads a CSV file with columns: city_id, x_coordinate, y_coordinate
    Returns a list of (city_id, x, y) tuples.
    """
    cities = []
    with open(filepath, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:  # skip empty rows
                continue
            city_id = row[0].strip()
            x_coord = float(row[1])
            y_coord = float(row[2])
            cities.append((city_id, x_coord, y_coord))
    return cities

def euclidean_distance(p1, p2):
    """
    Computes Euclidean distance between two points (x1, y1) and (x2, y2).
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def create_distance_matrix(cities):
    """
    Given a list of (city_id, x, y), create a 2D distance matrix where
    dist_matrix[i][j] = distance between city i and city j.
    """
    n = len(cities)
    dist_matrix = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = euclidean_distance((cities[i][1], cities[i][2]),
                                                       (cities[j][1], cities[j][2]))
    return dist_matrix

def total_tour_distance(tour, dist_matrix):
    """
    Computes the total distance of a given tour (list of city indices),
    assuming a round trip (last city connects back to the first).
    """
    distance = 0.0
    for i in range(len(tour)):
        distance += dist_matrix[tour[i]][tour[(i+1) % len(tour)]]
    return distance

def plot_tour(cities, tour, title="TSP Tour"):
    """
    Plots the TSP tour using matplotlib. 'tour' is a list of city indices in the visiting order.
    """
    x_coords = [cities[i][1] for i in tour]
    y_coords = [cities[i][2] for i in tour]

    # Close the loop
    x_coords.append(cities[tour[0]][1])
    y_coords.append(cities[tour[0]][2])

    plt.figure(figsize=(6,6))
    plt.plot(x_coords, y_coords, marker='o', linestyle='-')

    # Label each city
    for idx in tour:
        plt.text(cities[idx][1], cities[idx][2], cities[idx][0], fontsize=8, ha='right')

    plt.title(title)
    plt.show()