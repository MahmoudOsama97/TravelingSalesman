import csv
import random


def generate_cities_dataset(num_cities, filename="cities_10.csv", x_range=(0, 100), y_range=(0, 100)):
    """
    Generates a dataset of cities with random coordinates and saves it to a CSV file.

    Args:
        num_cities: The number of cities to generate.
        filename: The name of the CSV file to save the data to.
        x_range: A tuple (min, max) defining the range for x-coordinates.
        y_range: A tuple (min, max) defining the range for y-coordinates.
    """

    cities = []
    for i in range(num_cities):
        city_id = f"City{i+1}"  # Create city IDs like "City1", "City2", etc.
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        cities.append((city_id, x, y))

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # No header row needed, as per problem description
        for city in cities:
            writer.writerow(city)
    print(f"Generated {num_cities} cities and saved to {filename}")


def main():
    generate_cities_dataset(10)


if __name__ == "__main__":
    main()
