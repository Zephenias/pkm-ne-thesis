import matplotlib.pyplot as plt
import json

with open ("/home/zephenias/ThesisGit/pkm-ne-thesis/ThesisCode/sav/CNN_fitness_values_by_generation_total_20.json", "r") as file:
    data = json.load(file)

averages = []
min_values = []
max_values = []
meta = {"generations": 20, "max_steps": 100, "population_size": 20}

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
    ax.plot(x,averages, marker = 'o', label = 'Line Plot')
    plt.title('Average Fitness Value by Generation')
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness Value')
    description = f"Number of Generations: {meta['generations']}, \n Number of Playsteps {meta['max_steps']} \n Individuals/Generation past the initial one: {meta['population_size']}"
    fig.text(0.5, -0.1, description, fontsize = 10, va = 'top', ha = 'center')
    plt.grid(True)

    plt.figure()
    plt.plot(x, min_values, marker = 'o', label = 'Line Plot')
    plt.title('Minimum Fitness Value by Generation')
    plt.xlabel('Generation')
    plt.ylabel('Minimum Fitness Value')
    plt.text(0.5, 0.1, description, fontsize = 10, transform = plt.gcf().transFigure)
    plt.grid(True)

    plt.figure()
    plt.plot(x, max_values, marker = 'o', label = 'Line Plot')
    plt.title('Maximum Fitness Value by Generation')
    plt.xlabel('Generation')
    plt.ylabel('Maximum Fitness Value')
    plt.text(0.5, 0.1, description, fontsize = 10, transform = plt.gcf().transFigure)
    plt.grid(True)

    plt.show()

    return



if __name__ == "__main__":

    for generation in data:
        if generation != "metadata":
            averages.append(calculate_average(data[generation]))
            min_values.append(min(data[generation]))
            max_values.append(max(data[generation]))
        else:
            meta = generation
    create_plots()