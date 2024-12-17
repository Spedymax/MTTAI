from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import numpy as np

data = load_iris()
measurements = data.data[:, :2]
classes = data.target

flower_names = {1: 'Setosa', 2: 'Versicolour', 3: 'Virginica'}

centers, memberships, _, _, _, _, _ = fuzz.cluster.cmeans(
    measurements.T,
    3,
    3,
    error=0.001,
    maxiter=20,
    init=None
)

plt.title('Fuzzy clustering')
plt.scatter(measurements[classes == 0, 0], measurements[classes == 0, 1], c='purple', s=70)
plt.scatter(measurements[classes == 1, 0], measurements[classes == 1, 1], c='green', s=70)
plt.scatter(measurements[classes == 2, 0], measurements[classes == 2, 1], c='blue', s=70)
plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=100, c='red')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

clusters = np.argmax(memberships, axis=0)

point_colors = {5.1: 'pink', 7.0: 'olive', 5.5: 'cyan'}

plt.title('Original vs Clustered')
plt.scatter(measurements[classes == 0, 0], measurements[classes == 0, 1], c='purple', s=125)
plt.scatter(measurements[classes == 1, 0], measurements[classes == 1, 1], c='green', s=125)
plt.scatter(measurements[classes == 2, 0], measurements[classes == 2, 1], c='blue', s=125)

for i in range(3):
    cluster_points = measurements[clusters == i]
    plt.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        label=flower_names.get(i + 1),
        edgecolors='black',
        color=point_colors[cluster_points[0][0]]
    )

plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=100, c='red')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.legend()
plt.show()