import numpy as np
import matplotlib.pyplot as plt

M = np.array([
    [1, -1, -1, -1, 1],
    [1, 1, -1, 1, 1],
    [1, -1, 1, -1, 1],
    [1, -1, -1, -1, 1],
    [1, -1, -1, -1, 1],
    [1, -1, -1, -1, 1],
    [1, -1, -1, -1, 1]
])

A = np.array([
    [-1, 1, 1, 1, -1],
    [1, -1, -1, -1, 1],
    [1, -1, -1, -1, 1],
    [1, 1, 1, 1, 1],
    [1, -1, -1, -1, 1],
    [1, -1, -1, -1, 1],
    [1, -1, -1, -1, 1]
])

K = np.array([
    [1, -1, -1, -1, 1],
    [1, -1, -1, 1, -1],
    [1, -1, 1, -1, -1],
    [1, 1, -1, -1, -1],
    [1, -1, 1, -1, -1],
    [1, -1, -1, 1, -1],
    [1, -1, -1, -1, 1]
])

C = np.array([
    [-1, 1, 1, 1, -1],
    [1, -1, -1, -1, 1],
    [1, -1, -1, -1, -1],
    [1, -1, -1, -1, -1],
    [1, -1, -1, -1, -1],
    [1, -1, -1, -1, 1],
    [-1, 1, 1, 1, -1]
])


class HebbNetwork:
    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = np.zeros((input_size, input_size))

    def train(self, patterns):
        """Навчання мережі за правилом Хебба з підсиленим навчанням"""
        for pattern in patterns:
            flat_pattern = pattern.flatten()
            outer_product = np.outer(flat_pattern, flat_pattern)
            self.weights += 1.5 * outer_product

        max_weight = np.max(np.abs(self.weights))
        self.weights /= max_weight
        np.fill_diagonal(self.weights, 0)

    def recognize(self, pattern):
        """Розпізнавання вхідного патерну"""
        flat_pattern = pattern.flatten()
        output = np.sign(np.dot(self.weights, flat_pattern))
        return output.reshape(pattern.shape)

    def test_pattern(self, pattern, title="Test Pattern"):
        """Тестування патерну та візуалізація результату"""
        result = self.recognize(pattern)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(pattern, cmap='binary')
        ax1.set_title('Input Pattern')
        ax2.imshow(result, cmap='binary')
        ax2.set_title('Network Output')
        plt.suptitle(title)
        plt.show()


network = HebbNetwork(35)
patterns = [M, A, K, C]
network.train(patterns)


def add_noise(pattern, noise_level=0.1):
    noisy = pattern.copy()
    n_pixels = int(pattern.size * noise_level)
    coords = list(zip(*np.where(pattern != 0)))
    np.random.shuffle(coords)
    for i, j in coords[:n_pixels]:
        noisy[i, j] *= -1
    return noisy


for letter, pattern in zip(['M', 'A', 'K', 'C'], patterns):
    network.test_pattern(pattern, f'Testing Letter {letter}')

for letter, pattern in zip(['M', 'A', 'K', 'C'], patterns):
    noisy_pattern = add_noise(pattern, 0.1)
    network.test_pattern(noisy_pattern, f'Testing Noisy Letter {letter}')