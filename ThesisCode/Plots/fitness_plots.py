import matplotlib.pyplot as plt
import json

with open ("/home/zephenias/ThesisGit/pkm-ne-thesis/ThesisCode/Documenting/10ksteps_pop10_gen20_10cores/CNN_fitness_values_by_generation_total_200.json", "r") as file:
    data = json.load(file)

averages = []
min_values = []
max_values = []
meta = {"total_generations": 20, "max_steps": 100, "population_size": 20}

def calculate_average(value_list):
    sum_of_values = 0
    for value in value_list:
        sum_of_values += value
    return sum_of_values/len(value_list)

def create_plots():
    x = []
    for i in range (0,len(averages)):
        x.append(i)
    fig, ax = plt.subplots()
    ax.plot(x,averages, marker = "o", label = "Line Plot")
    plt.title(f"Average Fitness by Generation \n Gen: {meta['total_generations']} Steps: {meta['max_steps']} Individuals: {meta['population_size']}")
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness Value")
    plt.grid(True)

    plt.figure()
    plt.plot(x, min_values, marker = "o", label = "Line Plot")
    plt.title(f"Minimum Fitness Value by Generation \n Gen: {meta['total_generations']} Steps: {meta['max_steps']} Individuals: {meta['population_size']}")
    plt.xlabel("Generation")
    plt.ylabel("Minimum Fitness Value")
    plt.grid(True)

    plt.figure()
    plt.plot(x, max_values, marker = "o", label = "Line Plot")
    plt.title(f"Maximum Fitness Value by Generation \n Gen: {meta['total_generations']} Steps: {meta['max_steps']} Individuals: {meta['population_size']}")
    plt.xlabel("Generation")
    plt.ylabel("Maximum Fitness Value")
    plt.grid(True)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all three functions
    ax.plot(x, averages, label='Average Fitness Values', color='blue')
    ax.plot(x, max_values, label='Maximum Fitness Values', color='red')
    ax.plot(x, min_values, label='Minimum Fitness Values', color='green')

    ax.set_title(f"Layered view of Fitness Values \n Gen: {meta['total_generations']} Steps: {meta['max_steps']} Individuals: {meta['population_size']}")
    ax.set_xlabel("Generations")
    ax.set_ylabel("Fitness")
    ax.grid(True)

    plt.show()

    return



if __name__ == "__main__":

    for generation in data:
        if generation != "metadata":
            averages.append(calculate_average(data[generation]))
            min_values.append(min(data[generation]))
            max_values.append(max(data[generation]))
        else:
            meta["total_generations"] = data[generation]["total_generations"]
            meta["max_steps"] = data[generation]["max_steps"]
            meta["population_size"] = data[generation]["population_size"]
            print(meta)
    create_plots()