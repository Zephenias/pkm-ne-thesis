import matplotlib.pyplot as plt

if __name__ == "__main__":

    x = [1,2,3,4,6,7,9,14,15,20,30,41,55,56,60,96,112,113]
    conf_1_values = [
        0,0.5,0.666666666666667,0.5,0.333333333333333,0.285714285714286,0.333333333333333,0.214285714285714,0.2,0.15,0.1,0,0,0,0,0,0,0
                ]
    conf_2_values = [
        1,0.5,0.333333333333333,0.5,0.333333333333333,0.285714285714286,0.222222222222222,0.142857142857143,0.133333333333333,0.1,0.0666666666666667,0.0487804878048781,
        0.0363636363636364,0.0357142857142857,0.0333333333333333,0.0208333333333333,0.0267857142857143,0.0265486725663717
        ]
    conf_3_values = [
        0,0,0.333333333333333,0.25,0.166666666666667,0.142857142857143,0.222222222222222,0.214285714285714,0.266666666666667,0.2,0.133333333333333,
        0.121951219512195,0.0909090909090909,0.107142857142857,0.1,0,0,0
    ]
    conf_4_values = [
        1,1,0.666666666666667,0.5,0.333333333333333,0.428571428571429,0.333333333333333,0.214285714285714,0.2,0.15,0.133333333333333,0.0975609756097561,
        0.0909090909090909,0.0892857142857143,0.0833333333333333,0.0625,0.0535714285714286,0.0619469026548673
    ]
# Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all three functions
    ax.plot(x, conf_1_values, label='conf_CNN^1', color='blue')
    ax.plot(x, conf_2_values, label='conf_CNN^2', color='red')
    ax.plot(x, conf_3_values, label='conf_CNN^3', color='green')
    ax.plot(x, conf_4_values, label = 'conf_CNN^4', color = 'purple')

    ax.set_title(f"Comparison of found improvements/iterations \n Blue: conf¹, Red: conf², Green: conf³, Purple: conf⁴")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Improvements/iterations")
    ax.grid(True)

    plt.show()