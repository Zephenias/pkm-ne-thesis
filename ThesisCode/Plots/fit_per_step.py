import matplotlib.pyplot as plt

if __name__ == "__main__":

    x = [0,1,2,3,4,6,7,9,14,15,20,30,41,55,56,60,96,112,113]
    conf_1_values = [
        0.00822,0.00822,0.01176,0.01176,0.01176,0.01215,0.01215,0.01235,0.01235,0.01235,0.01235,0,0,0,0,0,0,0,0
        ]
    conf_2_values = [
        0.0133999999666667,0.0227,0.0227,0.0227,0.0245,0.0245,0.0245,0.0245,0.0245,0.0245,0.0245,0.0245,0.0245,0.0245,0.0245,0.0245,0.0245,0.0275,0.0275
        ]
    conf_3_values = [
        0.01452,0.01452,0.01452,0.0171,0.0171,0.0171,0.0171,0.01798,0.01877999998,0.0189,0.0189,0.0189,0.01902,0.01902,0.02154,0.02154,0,0,0
    ]
    conf_4_values = [
        0.0174,0.02115,0.027,0.027,0.027,0.027,0.03165,0.03165,0.03165,0.03165,0.03269995,0.03269995,0.03269995,0.0333,0.0333,0.0333,0.033445,0.033445,0.0336
    ]
# Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all three functions
    ax.plot(x, conf_1_values, label='conf_CNN^1', color='blue')
    ax.plot(x, conf_2_values, label='conf_CNN^2', color='red')
    ax.plot(x, conf_3_values, label='conf_CNN^3', color='green')
    ax.plot(x, conf_4_values, label = 'conf_CNN^4', color = 'purple')

    ax.set_title(f"Comparison of Fitness/Steps \n Blue: conf¹, Red: conf², Green: conf³, Purple: conf⁴")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Fitness/Steps")
    ax.grid(True)

    plt.show()