import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz


def visualize(x_data, ytext, plots):
    plt.figure(figsize=(10, 6))

    for p in plots:
        values = p['y']
        name = p.get('label', None)
        col = p.get('color', 'b')
        style = p.get('linestyle', '-')
        dots = p.get('marker', None)

        plt.plot(x_data, values, label=name, color=col, linestyle=style, marker=dots)

    plt.xlabel('x')
    plt.ylabel(ytext)
    plt.legend()
    plt.grid(True)
    plt.show()


def multi_plot(x, plots):
    n = len(plots)
    plt.figure(figsize=(15, 6))

    for i in range(n):
        plt.subplot(1, n, i + 1, title=plots[i].get('title', None))
        plt.plot(x, plots[i]['y'],
                 color=plots[i].get('color', 'b'),
                 linestyle=plots[i].get('linestyle', '-'),
                 marker=plots[i].get('marker', None))
        plt.xlabel('x')
        plt.grid(True)

    plt.tight_layout()
    plt.show()


x1 = np.linspace(-30, 30)

multi_plot(x1, [
    {'y': fuzz.trimf(x1, [-10, 5, 15]), 'title': 'Triangle MF [-10, 5, 15]'},
    {'y': fuzz.trapmf(x1, [-15, -5, 10, 20]), 'title': 'Trapezoid MF [-15, -5, 10, 20]'}
])

visualize(x1, 'Gauss MF', [
    {'y': fuzz.gaussmf(x1, 2, 7), 'label': 'Gauss MF [2, 7]'}
])

visualize(x1, 'Double Gauss MF', [
    {'y': fuzz.gauss2mf(x1, 3, 4, 8, 10), 'label': 'Gauss MF [3, 4, 8, 10]', 'color': 'purple'},
    {'y': fuzz.gauss2mf(x1, 5, 6, 12, 14), 'label': 'Gauss MF [5, 6, 12, 14]', 'linestyle': '--'}
])

visualize(x1, 'Bell MF', [
    {'y': fuzz.gbellmf(x1, 5, 9, 10), 'label': 'Bell MF [5, 9, 10]'}
])

x2 = np.linspace(0, 15)
multi_plot(x2, [
    {'y': fuzz.sigmf(x2, 5, 10), 'title': 'Sigmoid MF'},
    {'y': fuzz.dsigmf(x2, 7, 9, 11, 13), 'title': 'Sigmoid diff'},
    {'y': fuzz.psigmf(x2, 6, 8, 10, 12), 'title': 'Sigmoid prod'}
])

x3 = np.linspace(-25, 25)
g1 = fuzz.gaussmf(x3, 4, 6)
g2 = fuzz.gaussmf(x3, 8, 6)
visualize(x3, 'AND/OR on Gauss', [
    {'y': g1, 'label': 'Gauss 1', 'linestyle': '--'},
    {'y': g2, 'label': 'Gauss 2', 'linestyle': '--'},
    {'y': g1 * g2, 'label': 'AND'},
    {'y': g1 + g2 - g1 * g2, 'label': 'OR'}
])

x4 = np.linspace(-30, 30)
f = fuzz.gaussmf(x4, 2, 8)
visualize(x4, 'NOT operation', [
    {'y': f, 'label': 'Gauss MF'},
    {'y': 1 - f, 'label': 'NOT Gauss MF', 'color': 'orange', 'linestyle': '--'}
])