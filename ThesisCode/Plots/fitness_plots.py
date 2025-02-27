import matplotlib.pyplot as plt
import json

with open ("Documenting/CNN_Para_Reduced_SigmaT_Selection_Diff/2000steps_15pop_200gens_SigmaT_harshT30/selection_1/seed_46/strunzit_CNN_fitness_values_by_generation_total_200.json", "r") as file:
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
    ax.plot(x,averages, label = "Line Plot", color = "blue")
    plt.title(f"Average Fitness by Generation \n Iterations: {meta['total_generations']} Steps: {meta['max_steps']} Individuals: {meta['population_size']}")
    plt.xlabel("Iteration")
    plt.ylabel("Average Fitness Value")
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    plt.plot(x, min_values, label = "Line Plot", color = "blue")
    plt.title(f"Minimum Fitness Value by Generation \n Iterations: {meta['total_generations']} Steps: {meta['max_steps']} Individuals: {meta['population_size']}")
    plt.xlabel("Iteration")
    plt.ylabel("Minimum Fitness Value")
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    plt.plot(x, max_values, label = "Line Plot", color = "blue")
    plt.title(f"Maximum Fitness Value by Generation \n Iterations: {meta['total_generations']} Steps: {meta['max_steps']} Individuals: {meta['population_size']}")
    plt.xlabel("Iteration")
    plt.ylabel("Maximum Fitness Value")
    plt.grid(True)
    plt.tight_layout()

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all three functions
    ax.plot(x, averages, label='Average Fitness Values', color='blue')
    ax.plot(x, max_values, label='Maximum Fitness Values', color='red')
    ax.plot(x, min_values, label='Minimum Fitness Values', color='green')

    ax.set_title(f"Layered view of Fitness Values \n Iteration: {meta['total_generations']} Steps: {meta['max_steps']} Individuals: {meta['population_size']}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")
    ax.legend(title = "Legend", loc = "upper left", bbox_to_anchor = (1.0, 0.625))
    plt.subplots_adjust(right = 0.75)
    ax.grid(True)
    plt.tight_layout()



    return

def average_generation_plot(value_list, step):
    print(len(averages))
    x = []
    averages_gen = []
    chunk_counter = 1 
    for i in range (0, len(value_list), step):
        x.append(chunk_counter)
        chunk_counter += 1
        chunk = value_list[i:i+ step]
        avg = calculate_average(chunk)
        averages_gen.append(avg)
    
    plt.figure()
    plt.plot(x, averages_gen, label = "Line Plot", color = "purple", marker = "d", drawstyle = "steps-post" )
    plt.title(f"Average Fitness Value by Chunk of size {step} \n Iterations: {meta['total_generations']} Steps: {meta['max_steps']} Individuals: {meta['population_size']}")
    plt.xlabel("Chunk Number")
    plt.ylabel("Average Fitness Value of Chunk")
    plt.xticks(x)
    plt.grid(True)
    plt.tight_layout()


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
    average_generation_plot(averages, 10)
    
    
    plt.show()