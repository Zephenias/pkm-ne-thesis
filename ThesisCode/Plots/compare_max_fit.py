import matplotlib.pyplot as plt
import json

file_list = [
#conf 4    
"Documenting/CNN_Para_Reduced_SigmaT_Selection_Diff/2000steps_15pop_200gens_SigmaT_harshF300/selection_0/seed_42/ulexit_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Para_Reduced_SigmaT_Selection_Diff/2000steps_15pop_200gens_SigmaT_harshF300/selection_0/seed_43/stolzit_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Para_Reduced_SigmaT_Selection_Diff/2000steps_15pop_200gens_SigmaT_harshF300/selection_0/seed_44/terminus_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Para_Reduced_SigmaT_Selection_Diff/2000steps_15pop_200gens_SigmaT_harshF300/selection_0/seed_45/ananke_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Para_Reduced_SigmaT_Selection_Diff/2000steps_15pop_200gens_SigmaT_harshF300/selection_0/seed_46/unplugged_CNN_fitness_values_by_generation_total_200.json",
#conf 5
"Documenting/CNN_Para_Reduced_SigmaT_Selection_Diff/2000steps_15pop_200gens_SigmaT_harshF300/selection_1/seed_42/ulexit_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Para_Reduced_SigmaT_Selection_Diff/2000steps_15pop_200gens_SigmaT_harshF300/selection_1/seed_43/stolzit_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Para_Reduced_SigmaT_Selection_Diff/2000steps_15pop_200gens_SigmaT_harshF300/selection_1/seed_44/terminus_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Para_Reduced_SigmaT_Selection_Diff/2000steps_15pop_200gens_SigmaT_harshF300/selection_1/seed_45/ananke_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Para_Reduced_SigmaT_Selection_Diff/2000steps_15pop_200gens_SigmaT_harshF300/selection_1/seed_46/unplugged_CNN_fitness_values_by_generation_total_200.json",
#conf 6
"Documenting/CNN_Para_Reduced_SigmaT_Selection_Diff/2000steps_15pop_200gens_SigmaT_harshF300/selection_2/seed_42/ulexit_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Para_Reduced_SigmaT_Selection_Diff/2000steps_15pop_200gens_SigmaT_harshF300/selection_2/seed_43/stolzit_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Para_Reduced_SigmaT_Selection_Diff/2000steps_15pop_200gens_SigmaT_harshF300/selection_2/seed_44/terminus_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Para_Reduced_SigmaT_Selection_Diff/2000steps_15pop_200gens_SigmaT_harshF300/selection_2/seed_45/ananke_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Para_Reduced_SigmaT_Selection_Diff/2000steps_15pop_200gens_SigmaT_harshF300/selection_2/seed_46/unplugged_CNN_fitness_values_by_generation_total_200.json",
#conf 7
"Documenting/CNN_Parallel_Sigma_Mod/2000steps_15pop_200gens_harshF300/selection 0/seed 42/ananke_full_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Parallel_Sigma_Mod/2000steps_15pop_200gens_harshF300/selection 0/seed 43/gameboy_full_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Parallel_Sigma_Mod/2000steps_15pop_200gens_harshF300/selection 0/seed 44/obsidian_full_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Parallel_Sigma_Mod/2000steps_15pop_200gens_harshF300/selection 0/seed 45/olivenit_full_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Parallel_Sigma_Mod/2000steps_15pop_200gens_harshF300/selection 0/seed 46/stolzit_full_CNN_fitness_values_by_generation_total_200.json",
#conf 8
"Documenting/CNN_Parallel_Sigma_Mod/2000steps_15pop_200gens_harshF300/selection 1/seed 42/strunzit_full_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Parallel_Sigma_Mod/2000steps_15pop_200gens_harshF300/selection 1/seed 43/terminus_full_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Parallel_Sigma_Mod/2000steps_15pop_200gens_harshF300/selection 1/seed 44/trigger_full_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Parallel_Sigma_Mod/2000steps_15pop_200gens_harshF300/selection 1/seed 45/ulexit_full_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Parallel_Sigma_Mod/2000steps_15pop_200gens_harshF300/selection 1/seed 46/unplugged_full_CNN_fitness_values_by_generation_total_200.json",
#conf 9
"Documenting/CNN_Parallel_Sigma_Mod/2000steps_15pop_200gens_harshF300/selection 2/seed 42/ananke_full_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Parallel_Sigma_Mod/2000steps_15pop_200gens_harshF300/selection 2/seed 43/gameboy_full_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Parallel_Sigma_Mod/2000steps_15pop_200gens_harshF300/selection 2/seed 44/obsidian_full_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Parallel_Sigma_Mod/2000steps_15pop_200gens_harshF300/selection 2/seed 45/olivenit_full_CNN_fitness_values_by_generation_total_200.json",
"Documenting/CNN_Parallel_Sigma_Mod/2000steps_15pop_200gens_harshF300/selection 2/seed 46/stolzit_full_CNN_fitness_values_by_generation_total_200.json"
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

if len(file_list) >= 6:
    with open (file_list[5], "r") as file:
        data5 = json.load(file)

if len(file_list) >= 7:
    with open (file_list[6], "r") as file:
        data6 = json.load(file)

if len(file_list) >= 8:
    with open (file_list[7], "r") as file:
        data7 = json.load(file)

if len(file_list) >= 9:
    with open (file_list[8], "r") as file:
        data8 = json.load(file)

if len(file_list) >= 10:
    with open (file_list[9], "r") as file:
        data9 = json.load(file)

if len(file_list) >= 11:
    with open (file_list[10], "r") as file:
        data10 = json.load(file)

if len(file_list) >= 12:
    with open (file_list[11], "r") as file:
        data11 = json.load(file)

if len(file_list) >= 13:
    with open (file_list[12], "r") as file:
        data12 = json.load(file)

if len(file_list) >= 14:
    with open (file_list[13], "r") as file:
        data13 = json.load(file)

if len(file_list) >= 15:
    with open (file_list[14], "r") as file:
        data14 = json.load(file)

if len(file_list) >= 16:
    with open (file_list[15], "r") as file:
        data15 = json.load(file)

if len(file_list) >= 17:
    with open (file_list[16], "r") as file:
        data16 = json.load(file)

if len(file_list) >= 18:
    with open (file_list[17], "r") as file:
        data17 = json.load(file)

if len(file_list) >= 19:
    with open (file_list[18], "r") as file:
        data18 = json.load(file)

if len(file_list) >= 20:
    with open (file_list[19], "r") as file:
        data19 = json.load(file)

if len(file_list) >= 21:
    with open (file_list[20], "r") as file:
        data20 = json.load(file)

if len(file_list) >= 22:
    with open (file_list[21], "r") as file:
        data21 = json.load(file)

if len(file_list) >= 23:
    with open (file_list[22], "r") as file:
        data22 = json.load(file)

if len(file_list) >= 24:
    with open (file_list[23], "r") as file:
        data23 = json.load(file)

if len(file_list) >= 25:
    with open (file_list[24], "r") as file:
        data24 = json.load(file)

if len(file_list) >= 26:
    with open (file_list[25], "r") as file:
        data25 = json.load(file)

if len(file_list) >= 27:
    with open (file_list[26], "r") as file:
        data26 = json.load(file)

if len(file_list) >= 28:
    with open (file_list[27], "r") as file:
        data27 = json.load(file)

if len(file_list) >= 29:
    with open (file_list[28], "r") as file:
        data28 = json.load(file)

if len(file_list) >= 30:
    with open (file_list[29], "r") as file:
        data29 = json.load(file)

def calculate_average(value_list):
    sum_of_values = 0
    for value in value_list:
        sum_of_values += value
    return sum_of_values/len(value_list)

def calculate_average_list(list_of_value_lists):
    list = []
    for i in range (0, len(list_of_value_lists[0]), 1):
        element = calculate_average(
            [
                list_of_value_lists[0][i],
                list_of_value_lists[1][i],
                list_of_value_lists[2][i],
                list_of_value_lists[3][i],
                list_of_value_lists[4][i]
            ]
        )
        list.append(element)
    return list

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


def create_avg_plots(avg0, meta, avg1 = None, avg2 = None, avg3 = None, avg4 = None, avg5 = None):
# Create a figure and axis
    x = []
    for i in range (0,len(avg0)):
        x.append(i)
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all three functions
    ax.plot(x, avg0, label = "$conf_{CNN[42]}^7$", color='blue')
    if avg1:
        ax.plot(x, avg1, label = "$conf_{CNN[43]}^7$", color='red')
    if avg2:
        ax.plot(x, avg2, label = "$conf_{CNN[44]}^7$", color='green')
    if avg3:    
        ax.plot(x, avg3, label = "$conf_{CNN[45]}^7$", color = "purple")
    if avg4:
        ax.plot(x, avg4, label = "$conf_{CNN[46]}^7$", color = "orange")
    if avg5:
        ax.plot(x, avg5, label = "$conf_{cnn[46]}^6$", color = "black")

    ax.set_title(f"Layered view of Average Fitness Values \n Iteration: {meta['total_generations']} Steps: {meta['max_steps']} Individuals: {meta['population_size']}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")
    ax.legend(title = "Legend", loc = "upper left", bbox_to_anchor = (1.0, 0.625))
    plt.subplots_adjust(right = 0.75)
    ax.grid(True)
    plt.tight_layout()

def create_min_val_plots(min0, meta, min1 = None, min2 = None, min3 = None, min4 = None, min5 = None):
# Create a figure and axis
    x = []
    for i in range (0,len(min0)):
        x.append(i)
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all three functions
    ax.plot(x, min0, label = "$conf_{CNN[42]}^7$", color = "blue")
    if min1:
        ax.plot(x, min1, label = "$conf_{CNN[43]}^7$", color = "red")
    if min2:
        ax.plot(x, min2, label = "$conf_{CNN[44]}^7$", color = "green")
    if min3:
        ax.plot(x, min3, label = "$conf_{CNN[45]}^7$", color = "purple")
    if min4:
        ax.plot(x, min4, label = "$conf_{CNN[46]}^7$", color = "orange")
    if min5:
         ax.plot(x, min5, label = "$conf_{cnn[46]}^7$", color = "black")

    ax.set_title(f"Layered view of Minimum Fitness Values \n Iteration: {meta['total_generations']} Steps: {meta['max_steps']} Individuals: {meta['population_size']}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")
    ax.legend(title = "Legend", loc = "upper left", bbox_to_anchor = (1.0, 0.625))
    plt.subplots_adjust(right = 0.75)
    ax.grid(True)
    plt.tight_layout()

def create_max_val_plots(max0, meta, max1 = None, max2 = None, max3 = None, max4 = None,  max5 = None):
# Create a figure and axis
    x = []
    for i in range (0,len(max0)):
        x.append(i)
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all three functions
    ax.plot(x, max0, label = "$conf_{cnn[42]}^4$", color = "blue")
    if max1:
        ax.plot(x, max1, label = "$conf_{cnn[44]}^5$", color = "red")
    if max2:
        ax.plot(x, max2, label = "$conf_{cnn[43]}^6$", color = "green")
    if max3:    
        ax.plot(x, max3, label = "$conf_{CNN[46]}^7$", color = "purple")
    if max4:    
        ax.plot(x, max4, label = "$conf_{CNN[44]}^8$", color = "orange")
    if max5:
        ax.plot(x, max5, label = "$conf_{CNN[43]}^9$", color = "black")

    ax.set_title(f"Layered view of Maximum Fitness Values \n Iteration: {meta0['total_generations']} Steps: {meta0['max_steps']} Individuals: {meta0['population_size']}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")
    ax.legend(title = "Legend", loc = "upper left", bbox_to_anchor = (1.0, 0.625))
    plt.subplots_adjust(right = 0.75)
    ax.grid(True)
    plt.tight_layout()


def create_avg_max_val_plots(max0, meta, max1 = None, max2 = None, max3 = None, max4 = None,  max5 = None):
# Create a figure and axis
    x = []
    for i in range (0,len(max0)):
        x.append(i)
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all three functions
    ax.plot(x, max0, label = "$conf_{cnn[42,46]}^4$", color = "blue")
    if max1:
        ax.plot(x, max1, label = "$conf_{cnn[42,46]}^5$", color = "red")
    if max2:
        ax.plot(x, max2, label = "$conf_{cnn[42,46]}^6$", color = "green")
    if max3:    
        ax.plot(x, max3, label = "$conf_{CNN[42,46]}^7$", color = "purple")
    if max4:    
        ax.plot(x, max4, label = "$conf_{CNN[42,46]}^8$", color = "orange")
    if max5:
        ax.plot(x, max5, label = "$conf_{CNN[42,46]}^9$", color = "black")

    ax.set_title(f"Layered view of Average Maximum Fitness Values over all seeds \n Iteration: {meta0['total_generations']} Steps: {meta0['max_steps']} Individuals: {meta0['population_size']}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")
    ax.legend(title = "Legend", loc = "upper left", bbox_to_anchor = (1.0, 0.625))
    plt.subplots_adjust(right = 0.75)
    ax.grid(True)
    plt.tight_layout()

if __name__ == "__main__":
    

    if len(file_list) >= 30:
        average29, min_val29, max_val29, meta29 = value_extraction(data29)
        average28, min_val28, max_val28, meta28 = value_extraction(data28)
        average27, min_val27, max_val27, meta27 = value_extraction(data27)
        average26, min_val26, max_val26, meta26 = value_extraction(data26)
        average25, min_val25, max_val25, meta25 = value_extraction(data25)
        average24, min_val24, max_val24, meta24 = value_extraction(data24)
        average23, min_val23, max_val23, meta23 = value_extraction(data23)
        average22, min_val22, max_val22, meta22 = value_extraction(data22)
        average21, min_val21, max_val21, meta21 = value_extraction(data21)
        average20, min_val20, max_val20, meta20 = value_extraction(data20)
        average19, min_val19, max_val19, meta19 = value_extraction(data19)
        average18, min_val18, max_val18, meta18 = value_extraction(data18)
        average17, min_val17, max_val17, meta17 = value_extraction(data17)
        average16, min_val16, max_val16, meta16 = value_extraction(data16)
        average15, min_val15, max_val15, meta15 = value_extraction(data15)
        average14, min_val14, max_val14, meta14 = value_extraction(data14)
        average13, min_val13, max_val13, meta13 = value_extraction(data13)
        average12, min_val12, max_val12, meta12 = value_extraction(data12)
        average11, min_val11, max_val11, meta11 = value_extraction(data11)
        average10, min_val10, max_val10, meta10 = value_extraction(data10)

    if len(file_list) >= 10:
        average9, min_val9, max_val9, meta9 = value_extraction(data9)
        average8, min_val8, max_val8, meta8 = value_extraction(data8)
        average7, min_val7, max_val7, meta7 = value_extraction(data7)
        average6, min_val6, max_val6, meta6 = value_extraction(data6)
    if len(file_list) >= 6:
        average5, min_val5, max_val5, meta5 = value_extraction(data5)
    if len(file_list) >= 5:
        average4, min_val4, max_val4, meta4 = value_extraction(data4)
    if len(file_list) >= 4:
        average3, min_val3, max_val3, meta3 = value_extraction(data3)
    if len(file_list) >= 3:
        average2, min_val2, max_val2, meta2 = value_extraction(data2)
    if len(file_list) >= 2:
        average1, min_val1, max_val1, meta1 = value_extraction(data1)
    if len(file_list) >= 1:
        average0, min_val0, max_val0, meta0 = value_extraction(data0)
    else:
        print("The fuck are you doing?")
    
    if len(file_list) == 6:
        create_avg_plots(average0, meta0, average1, average2, average3, average4, average5)
        create_min_val_plots(min_val0, meta0, min_val1, min_val2, min_val3, min_val4, min_val5)
        create_max_val_plots(max_val0, meta0, max_val1, max_val2, max_val3, max_val4, max_val5)
    elif len(file_list) == 5:
        create_avg_plots(average0, meta0, average1, average2, average3, average4)
        create_min_val_plots(min_val0, meta0, min_val1, min_val2, min_val3, min_val4)
        create_max_val_plots(max_val0, meta0, max_val1, max_val2, max_val3, max_val4)
    elif len(file_list) == 10:
        avg_max_val1 = calculate_average_list([max_val0, max_val1, max_val2, max_val3, max_val4])
        avg_max_val2 = calculate_average_list([max_val5, max_val6, max_val7, max_val8, max_val9])
        create_max_val_plots(avg_max_val1, meta0, avg_max_val2, max_val0, max_val9)
        create_avg_max_val_plots(avg_max_val1, meta0, avg_max_val2)
    elif len(file_list) == 30:
        avg_max_val1 = calculate_average_list([max_val0, max_val1, max_val2, max_val3, max_val4]) #conf cnn 4
        avg_max_val2 = calculate_average_list([max_val5, max_val6, max_val7, max_val8, max_val9]) #conf cnn 5
        avg_max_val3 = calculate_average_list([max_val10, max_val11, max_val12, max_val13, max_val14])#conf cnn 6
        avg_max_val4 = calculate_average_list([max_val15, max_val16, max_val17, max_val18, max_val19])#conf CNN 7
        avg_max_val5 = calculate_average_list([max_val20, max_val21, max_val22, max_val23, max_val24])#conf CNN 8
        avg_max_val6 = calculate_average_list([max_val25, max_val26, max_val27, max_val28, max_val29])#conf CNN 9
        create_avg_max_val_plots(avg_max_val1, meta0, avg_max_val2, avg_max_val3, avg_max_val4, avg_max_val5, avg_max_val6) #compares all selections of cnn and CNN
        create_max_val_plots(max_val0, meta0, max_val7, max_val11, max_val19,max_val22, max_val26) #compares best indivduals from cnn and CNN over all selections ans seeds
    plt.show()
