import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

def gauss(x, c, sigma):
    return np.exp(-0.5 * ((x - c) / sigma) ** 2)

def humidity_memberships(h):
    return {
        'low': gauss(h, 20, 10),
        'medium': gauss(h, 45, 12),
        'high': gauss(h, 70, 12)
    }

def temperature_memberships(t):
    return {
        'cold': gauss(t, 12, 3.5),
        'comfortable': gauss(t, 20, 4),
        'hot': gauss(t, 28, 3.5)
    }

def price_memberships(p):
    return {
        'cheap': gauss(p, 0.25, 0.08),
        'normal': gauss(p, 0.5, 0.08),
        'expensive': gauss(p, 0.75, 0.08)
    }

INTENSITY_X = np.linspace(0, 100, 400)
INTENSITY_SETS = {
    'low': gauss(INTENSITY_X, 10, 10),
    'medium': gauss(INTENSITY_X, 45, 12),
    'high': gauss(INTENSITY_X, 80, 12)
}

WEIGHTS = {
    'humidity': 1.0,
    'temperature': 0.5,
    'price': 0.8
}

def infer_intensity(h, t, p):
    h_m = humidity_memberships(h)
    t_m = temperature_memberships(t)
    p_m = price_memberships(p)

    aggregated = np.zeros_like(INTENSITY_X)

    rules = [
        ('low', 'cold', 'cheap', 'high', 1.0),
        ('low', 'comfortable', 'normal', 'high', 0.9),
        ('low', 'hot', 'expensive', 'medium', 0.7),
        ('medium', 'cold', 'cheap', 'medium', 0.9),
        ('medium', 'comfortable', 'normal', 'medium', 0.8),
        ('medium', 'hot', 'expensive', 'low', 0.6),
        ('high', 'cold', 'cheap', 'medium', 0.7),
        ('high', 'comfortable', 'normal', 'low', 0.9),
        ('high', 'hot', 'expensive', 'low', 1.0),
    ]

    for (h_label, t_label, p_label, out_label, base_strength) in rules:
        mu_h = h_m[h_label] * WEIGHTS['humidity']
        mu_t = t_m[t_label] * WEIGHTS['temperature']
        mu_p = p_m[p_label] * WEIGHTS['price']

        strength = base_strength * mu_h * mu_t * mu_p
        if strength <= 1e-8:
            continue

        out_mu = INTENSITY_SETS[out_label] * strength
        aggregated = np.maximum(aggregated, out_mu)

    denom = np.sum(aggregated)
    if denom < 1e-8:
        return 0.0

    return float(np.sum(aggregated * INTENSITY_X) / denom)

def compute_surface(price, H_steps=60, T_steps=60):
    H = np.linspace(0, 100, H_steps)
    T = np.linspace(10, 30, T_steps)
    HH, TT = np.meshgrid(H, T)
    ZZ = np.zeros_like(HH)

    for i in range(HH.shape[0]):
        for j in range(HH.shape[1]):
            ZZ[i, j] = infer_intensity(HH[i, j], TT[i, j], price)
    return HH, TT, ZZ

def plot_all_prices(prices):
    fig = plt.figure(figsize=(18, 6))

    for idx, p in enumerate(prices, start=1):
        HH, TT, ZZ = compute_surface(p)
        ax = fig.add_subplot(1, 3, idx, projection='3d')
        surf = ax.plot_surface(HH, TT, ZZ, cmap='viridis', edgecolor='none', alpha=0.9)
        ax.set_xlabel("Humidity (%)")
        ax.set_ylabel("Temperature (°C)")
        ax.set_zlabel("Intensity (%)")
        ax.set_title(f"Price = {p:.2f}")
        fig.colorbar(surf, ax=ax, shrink=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    prices = [0.25, 0.5, 0.75]
    plot_all_prices(prices)
