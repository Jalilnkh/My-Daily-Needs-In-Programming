# The code is written by Jalil Nourmohammadi Khiarak and all copy rights is reserved.
import matplotlib.pyplot as plt

def plot_kmean_input_data(X):
    plt.scatter(X[:, 0], X[:, 1], s=50, c='blue')
    plt.title("Input Data for K-means++")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid()
    plt.show()  