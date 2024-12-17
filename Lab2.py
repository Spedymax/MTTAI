import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

x_range = np.arange(1, 7, 0.1)
y_range = np.cos(x_range) / x_range - np.sin(x_range) / x_range ** 2
z_range = np.sin(x_range / 2) + y_range * np.sin(x_range)

def make_plot(title, x_name, y_name, arrays):
    plt.figure(figsize=(10, 6))
    for arr in arrays:
        plt.plot(x_range, arr['y'], label=arr['label'])
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

make_plot('Function y', 'x', 'y', [
    {'y': y_range, 'label': 'y = cos(x)/x - sin(x)/x^2'}
])

make_plot('Function z', 'x', 'z', [
    {'y': z_range, 'label': 'z = sin(x/2) + y*sin(x)'}
])

x = ctrl.Antecedent(x_range, 'x')
y = ctrl.Antecedent(np.arange(y_range.min(), y_range.max(), 0.1), 'y')
z = ctrl.Consequent(np.arange(z_range.min(), z_range.max(), 0.1), 'z')

def get_ranges(vals, n):
    step = (vals.max() - vals.min()) / n
    ranges = []
    for i in range(n):
        ranges.append((vals.min() + i * step, vals.min() + (i + 1) * step))
    return ranges

def set_gauss(var, ranges, names):
    for r, n in zip(ranges, names):
        var[n] = fuzz.gaussmf(var.universe, r[1], 0.1)

def set_bell(var, ranges, names):
    for r, n in zip(ranges, names):
        var[n] = fuzz.gbellmf(var.universe, 0.1, 5, r[1])

x_ranges = get_ranges(x_range, 6)
y_ranges = get_ranges(y_range, 6)
z_ranges = get_ranges(z_range, 9)

names_x = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
names_y = ['y1', 'y2', 'y3', 'y4', 'y5', 'y6']
names_z = ['z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9']

set_gauss(x, x_ranges, names_x)
set_gauss(y, y_ranges, names_y)
set_gauss(z, z_ranges, names_z)

set_bell(x, x_ranges, names_x)
set_bell(y, y_ranges, names_y)
set_bell(z, z_ranges, names_z)

x.view()
y.view()
z.view()

rules = []
for i in range(6):
    for j in range(6):
        r = ctrl.Rule(x[names_x[i]] & y[names_y[j]], z[names_z[(i + j) % 9]])
        rules.append(r)

sys = ctrl.ControlSystem(rules)
sim = ctrl.ControlSystemSimulation(sys)

output = []
for xv, yv in zip(x_range, y_range):
    sim.input['x'] = xv
    sim.input['y'] = yv
    sim.compute()
    output.append(sim.output['z'])

make_plot('Function vs Simulation', 'x', 'z', [
    {'y': z_range, 'label': 'Function z'},
    {'y': output, 'label': 'Simulation z'}
])