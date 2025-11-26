import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gauss(x, c, sigma):
    return np.exp(-0.5 * ((x - c) / sigma) ** 2)

humid_terms = {
    'low':    {'c': 20, 's': 12},
    'medium': {'c': 50, 's': 10},
    'high':   {'c': 80, 's': 12},
}

temp_terms = {
    'cold': {'c': 10, 's': 6},
    'ok':   {'c': 22, 's': 6},
    'hot':  {'c': 34, 's': 6},
}

tariff_terms = {
    'cheap':    {'c': 0.2, 's': 0.12},
    'normal':   {'c': 0.6, 's': 0.12},
    'expensive':{'c': 1.2, 's': 0.18},
}

out_terms = {
    'off':    {'c': 0,  's': 6},
    'low':    {'c': 25, 's': 8},
    'medium': {'c': 50, 's': 10},
    'high':   {'c': 85, 's': 8},
}

weight_hum = 1.0
weight_tariff = 0.6
weight_temp = 0.3

rules = []
base_from_hum = {'low':'high', 'medium':'medium', 'high':'low'}
out_levels = ['off','low','medium','high']

def shift_level(level, shift):
    idx = out_levels.index(level)
    idx = max(0, min(len(out_levels)-1, idx + shift))
    return out_levels[idx]

for h in humid_terms.keys():
    for t in temp_terms.keys():
        for ta in tariff_terms.keys():
            base = base_from_hum[h]
            if ta == 'cheap':
                base = shift_level(base, +1)
            elif ta == 'expensive':
                base = shift_level(base, -1)
            rules.append(((h,t,ta), base))

def fuzzify_humidity(h):
    return {name: gauss(h, p['c'], p['s']) * weight_hum for name,p in humid_terms.items()}

def fuzzify_temp(T):
    return {name: gauss(T, p['c'], p['s']) * weight_temp for name,p in temp_terms.items()}

def fuzzify_tariff(price):
    return {name: gauss(price, p['c'], p['s']) * weight_tariff for name,p in tariff_terms.items()}

def output_mfs(x):
    return {name: gauss(x, p['c'], p['s']) for name,p in out_terms.items()}

x_out = np.linspace(0,100,801)
out_mf_cache = output_mfs(x_out)

def infer(h_val, t_val, tar_val):
    fh = fuzzify_humidity(h_val)
    ft = fuzzify_temp(t_val)
    ftar = fuzzify_tariff(tar_val)
    agg = np.zeros_like(x_out)
    for (h_term,t_term,tar_term), out_term in rules:
        activation = min(fh[h_term], ft[t_term], ftar[tar_term])
        if activation <= 0:
            continue
        mf_out = out_mf_cache[out_term]
        clipped = np.minimum(activation, mf_out)
        agg = np.maximum(agg, clipped)
    denom = np.sum(agg)
    if denom == 0:
        crisp = 0.0
    else:
        crisp = np.sum(agg * x_out) / denom
    return crisp, agg

tariff_values = [0.18, 0.6, 1.2]
tariff_labels = ['tania (0.18)', 'normalna (0.6)', 'droga (1.2)']

H_vals = np.linspace(0,100,81)
T_vals = np.linspace(0,40,41)
H_grid, T_grid = np.meshgrid(H_vals, T_vals)

results = {}
for tar, label in zip(tariff_values, tariff_labels):
    Z = np.zeros_like(H_grid)
    for i in range(H_grid.shape[0]):
        for j in range(H_grid.shape[1]):
            h = H_grid[i,j]
            t = T_grid[i,j]
            z,_ = infer(h,t,tar)
            Z[i,j] = z
    results[label] = Z

for label, Z in results.items():
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111, projection='3d')
    X = H_grid
    Y = T_grid
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
    ax.set_xlabel('Wilgotność [%]')
    ax.set_ylabel('Temperatura [°C]')
    ax.set_zlabel('Moc nawilżacza [%]')
    ax.set_title(f'Charakterystyka — taryfa: {label}')
    ax.view_init(elev=25, azim=-120)
    plt.show()
