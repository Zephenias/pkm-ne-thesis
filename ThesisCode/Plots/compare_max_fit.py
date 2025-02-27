import matplotlib.pyplot as plt
import json

file_list = [
"Documenting/CNN_Para_Reduced_SigmaT_Selection_Diff/2000steps_15pop_200gens_SigmaT_harshT30/selection_1/seed_42/gameboy_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Para_Reduced_SigmaT_Selection_Diff/2000steps_15pop_200gens_SigmaT_harshT30/selection_1/seed_43/obsidian_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Para_Reduced_SigmaT_Selection_Diff/2000steps_15pop_200gens_SigmaT_harshT30/selection_1/seed_44/trigger_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Para_Reduced_SigmaT_Selection_Diff/2000steps_15pop_200gens_SigmaT_harshT30/selection_1/seed_45/olivenit_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Para_Reduced_SigmaT_Selection_Diff/2000steps_15pop_200gens_SigmaT_harshT30/selection_1/seed_46/strunzit_CNN_fitness_values_by_generation_total_200.json"
]


with open (file_list[0], "r") as file:
    data0 = json.load(file)

with open (file_list[1], "r") as file:
    data1 = json.load(file)

with open (file_list[2], "r") as file:
    data2 = json.load(file)

with open (file_list[3], "r") as file:
    data3 = json.load(file)

with open (file_list[4], "r") as file:
    data4 = json.load(file)

def calculate_average(value_list):
    sum_of_values = 0
    for value in value_list:
        sum_of_values += value
    return sum_of_values/len(value_list)


def value_extraction(data):
    averages = []
    min_values = []
    max_values = []
    meta = {}
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
    return averages, min_values, max_values, meta

def create_avg_plots(avg0, avg1, avg2, avg3, avg4, meta):
# Create a figure and axis
    x = []
    for i in range (0,len(avg0)):
        x.append(i)
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all three functions
    ax.plot(x, avg0, label = "$conf_{cnn[42]}^2$", color='blue')
    ax.plot(x, avg1, label = "$conf_{cnn[43]}^2$", color='red')
    ax.plot(x, avg2, label = "$conf_{cnn[44]}^2$", color='green')
    ax.plot(x, avg3, label = "$conf_{cnn[45]}^2$", color = "purple")
    ax.plot(x, avg4, label = "$conf_{cnn[46]}^2$", color = "orange")

    ax.set_title(f"Layered view of Average Fitness Values \n Iteration: {meta['total_generations']} Steps: {meta['max_steps']} Individuals: {meta['population_size']}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")
    ax.legend(title = "Legend", loc = "upper left", bbox_to_anchor = (1.0, 0.625))
    plt.subplots_adjust(right = 0.75)
    ax.grid(True)
    plt.tight_layout()

def create_min_val_plots(min0, min1, min2, min3, min4, meta):
# Create a figure and axis
    x = []
    for i in range (0,len(min0)):
        x.append(i)
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all three functions
    ax.plot(x, min0, label = "$conf_{cnn[42]}^3$", color = "blue")
    ax.plot(x, min1, label = "$conf_{cnn[43]}^3$", color = "red")
    ax.plot(x, min2, label = "$conf_{cnn[44]}^3$", color = "green")
    ax.plot(x, min3, label = "$conf_{cnn[45]}^3$", color = "purple")
    ax.plot(x, min4, label = "$conf_{cnn[46]}^3$", color = "orange")

    ax.set_title(f"Layered view of Minimum Fitness Values \n Iteration: {meta['total_generations']} Steps: {meta['max_steps']} Individuals: {meta['population_size']}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")
    ax.legend(title = "Legend", loc = "upper left", bbox_to_anchor = (1.0, 0.625))
    plt.subplots_adjust(right = 0.75)
    ax.grid(True)
    plt.tight_layout()

def create_max_val_plots(max0, max1, max2, max3, max4, meta):
# Create a figure and axis
    x = []
    for i in range (0,len(max0)):
        x.append(i)
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all three functions
    ax.plot(x, max0, label = "$conf_{cnn[42]}^3$", color = "blue")
    ax.plot(x, max1, label = "$conf_{cnn[43]}^3$", color = "red")
    ax.plot(x, max2, label = "$conf_{cnn[44]}^3$", color = "green")
    ax.plot(x, max3, label = "$conf_{cnn[45]}^3$", color = "purple")
    ax.plot(x, max4, label = "$conf_{cnn[46]}^3$", color = "orange")

    ax.set_title(f"Layered view of Maximum Fitness Values \n Iteration: {meta0['total_generations']} Steps: {meta0['max_steps']} Individuals: {meta0['population_size']}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")
    ax.legend(title = "Legend", loc = "upper left", bbox_to_anchor = (1.0, 0.625))
    plt.subplots_adjust(right = 0.75)
    ax.grid(True)
    plt.tight_layout()

if __name__ == "__main__":
    
    average0, min_val0, max_val0, meta0 = value_extraction(data0)
    average1, min_val1, max_val1, meta1 = value_extraction(data1)
    average2, min_val2, max_val2, meta2 = value_extraction(data2)
    average3, min_val3, max_val3, meta3 = value_extraction(data3)
    average4, min_val4, max_val4, meta4 = value_extraction(data4)

    create_avg_plots(average0, average1, average2, average3, average4, meta0)
    create_min_val_plots(min_val0, min_val1, min_val2, min_val3, min_val4, meta0)
    create_max_val_plots(max_val0, max_val1, max_val2, max_val3, max_val4, meta0)

    plt.show()
