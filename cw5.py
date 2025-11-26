import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def gauss(x, c, sigma):
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))



def humidity_sets(x):
    return {
        "low": gauss(x, 20, 10),
        "medium": gauss(x, 50, 10),
        "high": gauss(x, 80, 10)
    }

def temp_sets(x):
    return {
        "cold": gauss(x, 15, 3),
        "normal": gauss(x, 22, 3),
        "hot": gauss(x, 28, 3)
    }

def price_sets(x):
    return {
        "cheap": gauss(x, 0.2, 0.15),
        "mid": gauss(x, 0.5, 0.15),
        "expensive": gauss(x, 0.8, 0.15)
    }



def humidifier_sets(x):
    return {
        "low": gauss(x, 10, 10),
        "medium": gauss(x, 40, 10),
        "high": gauss(x, 80, 10)
    }



def infer(hum, temp, price):
    rules = []
    
    rules.append(("high", hum["low"]))
    
    rules.append(("medium", hum["medium"]))

    rules.append(("low", hum["high"]))

    t_weight = 0.3
    rules.append(("high", t_weight * temp["hot"]))
    rules.append(("low",  t_weight * temp["cold"]))

    p_weight = 0.6
    rules.append(("high", p_weight * price["cheap"]))
    rules.append(("low",  p_weight * price["expensive"]))

    aggregated = {"low": 0, "medium": 0, "high": 0}

    for label, strength in rules:
        aggregated[label] = max(aggregated[label], strength)

    return aggregated


def defuzzify(agg):
    xs = np.linspace(0, 100, 500)
    output_mf = np.zeros_like(xs)

    sets_out = humidifier_sets(xs)

    for label in agg:
        output_mf = np.maximum(output_mf, np.minimum(agg[label], sets_out[label]))

    return np.sum(xs * output_mf) / np.sum(output_mf)


def humidifier_control(humidity, temp, price):
    hum = humidity_sets(humidity)
    t = temp_sets(temp)
    p = price_sets(price)

    agg = infer(hum, t, p)
    return defuzzify(agg)


humidity_vals = np.linspace(0, 100, 60)
temp_vals = np.linspace(10, 30, 60)

prices = [0.2, 0.5, 0.8]  

fig = plt.figure(figsize=(16, 5))

for i, price in enumerate(prices, 1):
    H, T = np.meshgrid(humidity_vals, temp_vals)

    Z = np.zeros_like(H)

    for ix in range(H.shape[0]):
        for jx in range(H.shape[1]):
            Z[ix, jx] = humidifier_control(H[ix, jx], T[ix, jx], price)

    ax = fig.add_subplot(1, 3, i, projection='3d')
    ax.plot_surface(H, T, Z, cmap="viridis")
    ax.set_title(f"Cena prądu = {price}")
    ax.set_xlabel("Wilgotność (%)")
    ax.set_ylabel("Temperatura (°C)")
    ax.set_zlabel("Intensywność nawilżania (%)")
    ax.view_init(30, 230)

plt.suptitle("Charakterystyka działania sterownika nawilżacza (logika rozmyta)", fontsize=16)
plt.tight_layout()
plt.show()
