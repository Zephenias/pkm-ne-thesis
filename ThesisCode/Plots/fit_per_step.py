import matplotlib.pyplot as plt

if __name__ == "__main__":

    x = [
        0,1,2,3,4,6,7,9,10,11,14,15,18,20,29,30,32,37,41,55,56,60,65,87,96,112,113,117
        ]
    conf_1_values = [
        0.00822,0.00822,0.01176,0.01176,0.01176,0.01215,0.01215,0.01235,0.01235,0.01235,0.01235,0.01235,0.01235,0.01235
                    ]
    conf_2_values = [
       0.0133999999666667,0.0227,0.0227,0.0227,0.0245,0.0245,0.0245,0.0245,0.0245,0.0245,0.0245,0.0245,0.0245,0.0245,0.0245,0.0245,0.0245,0.0245,0.0245,0.0245,0.0245,0.0245,
       0.0245,0.0245,0.0245,0.0275,0.0275,0.0275
                    ]
    conf_3_values = [
        0.01452,0.01452,0.01452,0.0171,0.0171,0.0171,0.0171,0.01798,0.01798,0.01798,0.01877999998,0.0189,0.0189,0.0189,0.0189,0.0189,0.0189,0.0189,0.01902,0.01902,0.02154,
        0.02154
                    ]
    conf_4_values = [
        0.0174,0.02115,0.027,0.027,0.027,0.027,0.03165,0.03165,0.03165,0.03165,0.03165,0.03165,0.03165,0.03165,0.03165,0.03269995,0.03269995,0.03269995,0.03269995,0.0333,0.0333,
        0.0333,0.0333,0.0333,0.033445,0.033445,0.0336,0.0336
                    ]
    conf_5_values = [
        0.01452,0.01452,0.01452,0.01452,0.01452,0.0171,0.0171,0.0171,0.0171,0.0171,0.0171,0.0171,0.01798,0.01798,0.0187798,0.0187798,0.0189,0.0189,0.0189,0.0189,0.0189,0.0189,
        0.0189,0.01902,0.01902,0.01902,0.01902,0.02154
                    ]
    conf_6_values = [
        0.00957,0.0096,0.00996,0.00996,0.00996,0.00996,0.00996,0.00996,0.01025,0.01068,0.01068,0.01068,0.01068,0.01068,0.01068,0.01068,0.01068,0.0116399,0.0116399,0.0116399,
        0.0116399,0.0116399,0.0123899,0.0123899,0.0123899,0.0123899,0.0123899,0.0123899
                    ]

# Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all three functions
    ax.plot(x[:len(conf_1_values)], conf_1_values, label='$conf_{CNN}^1$', color='blue', drawstyle = "steps-pre", linestyle = ":", marker = "+")
    ax.plot(x, conf_2_values, label='$conf_{CNN}^2$', color='red', drawstyle = "steps-pre", linestyle = ":", marker = "+")
    ax.plot(x[:len(conf_3_values)], conf_3_values, label='$conf_{CNN}^3$', color='green', drawstyle = "steps-pre", linestyle = ":", marker = "+")
    ax.plot(x, conf_4_values, label = '$conf_{CNN}^4$', color = 'purple',drawstyle = "steps-pre", linestyle = ":", marker = "+")
    ax.plot(x, conf_5_values, label = '$conf_{CNN}^5$ ', color = 'orange', drawstyle = "steps-pre", linestyle = ":", marker = "+")
    ax.plot(x, conf_6_values, label = '$conf_{CNN}^6$', color = 'black', drawstyle = "steps-pre", linestyle = ":", marker = "+")

    ax.set_title(f"Comparison of Fitness/Steps")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Fitness/Steps")
    ax.legend(title = "Legend", loc = "upper left", bbox_to_anchor = (1.0, 0.625))
    ax.grid(True)
    plt.tight_layout()

    plt.show()