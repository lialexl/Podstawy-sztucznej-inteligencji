import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import random

random.seed(0)
np.random.seed(0)

x = sp.symbols('x')
func_input = input("Wprowadź funkcję y(x) (np. x**2 - 10*cos(2*pi*x) + x/5):\n")
domain_input = input("Wprowadź przedział (np. -5,5):\n")
ext_type = input("Wybierz ekstremum ('min' dla minimum, 'max' dla maksimum):\n").strip().lower()
alg_choice = input("Wybierz algorytm ('hill' dla hill climbing, 'sa' dla symulowane wyżarzanie):\n").strip().lower()

try:
    a_str, b_str = domain_input.replace('[', '').replace(']', '').split(',')
    a, b = float(a_str), float(b_str)
except Exception:
    print("Błąd wczytania przedziału, ustawiam domyślnie [-5, 5].")
    a, b = -5.0, 5.0

try:
    expr = sp.sympify(func_input, locals={'pi': sp.pi})
except Exception:
    print("Błąd wczytania funkcji, ustawiam domyślnie y = x**2.")
    expr = x**2

f_np = sp.lambdify(x, expr, modules=['numpy', 'math'])

if ext_type not in ('min', 'max'):
    ext_type = 'min'
if alg_choice not in ('hill', 'sa'):
    alg_choice = 'hill'

iters = max(1, int(input("Liczba iteracji (np. 500):\n") or "500"))
step_init = (b - a) / 50.0
positions = []

def clip(v):
    return max(min(v, b), a)

def eval_obj(xx):
    try:
        return float(f_np(xx))
    except Exception:
        return float(sp.N(expr.subs(x, xx)))

def hill_climb(x0, iters, step):
    pos = []
    xcur = x0
    pos.append((xcur, f_np(xcur)))
    for i in range(iters):
        s = step * (1 - i / iters) + 1e-9
        cand1 = clip(xcur + s)
        cand2 = clip(xcur - s)
        ycur = f_np(xcur)
        y1 = f_np(cand1)
        y2 = f_np(cand2)
        if ext_type == 'min':
            if y1 < ycur and y1 <= y2:
                xcur = cand1
            elif y2 < ycur and y2 <= y1:
                xcur = cand2
        else:
            if y1 > ycur and y1 >= y2:
                xcur = cand1
            elif y2 > ycur and y2 >= y1:
                xcur = cand2
        pos.append((xcur, f_np(xcur)))
    return xcur, pos

def simulated_annealing(x0, iters, T0, cooling, param):
    pos = []
    xcur = x0
    T = T0
    pos.append((xcur, f_np(xcur)))
    for i in range(iters):
        step = step_init * (1 - i / iters) + 1e-6
        xnew = clip(xcur + random.uniform(-step, step))
        ycur = f_np(xcur)
        ynew = f_np(xnew)
        delta = ynew - ycur
        accept = False
        if (ext_type == 'min' and delta < 0) or (ext_type == 'max' and delta > 0):
            accept = True
        else:
            if T > 1e-12:
                p = math.exp(-abs(delta) / T)
                if random.random() < p:
                    accept = True
        if accept:
            xcur = xnew
        pos.append((xcur, f_np(xcur)))
        if cooling == 'geom':
            alpha = param
            T = T * alpha
        else:
            T = max(0.0, T - param)
    return xcur, pos

positions.clear()
best_x, best_y, best_path = None, float('inf') if ext_type == 'min' else -float('inf'), []

if alg_choice == 'hill':
    for _ in range(20):  
        x0 = random.uniform(a, b)
        x_candidate, path = hill_climb(x0, iters, step_init)
        y_candidate = f_np(x_candidate)
        if (ext_type == 'min' and y_candidate < best_y) or (ext_type == 'max' and y_candidate > best_y):
            best_x, best_y, best_path = x_candidate, y_candidate, path

else:
    T0 = float(input("Początkowa temperatura T0 (np. 1.0):\n") or "1.0")
    cooling = input("Strategia ochładzania ('geom' geometryczna, 'lin' liniowa):\n").strip().lower()
    if cooling == 'geom':
        alpha = float(input("Współczynnik alfa (np. 0.99):\n") or "0.99")
        param = alpha
    else:
        dec = float(input("Spadek temperatury na krok (np. T0/iters):\n") or str(T0/iters))
        param = dec
        cooling = 'lin'

    for _ in range(20):
        x0 = random.uniform(a, b)
        x_candidate, path = simulated_annealing(x0, iters, T0, cooling, param)
        y_candidate = f_np(x_candidate)
        if (ext_type == 'min' and y_candidate < best_y) or (ext_type == 'max' and y_candidate > best_y):
            best_x, best_y, best_path = x_candidate, y_candidate, path

res_x, res_y = best_x, best_y
positions = best_path  


print("\n=== Wynik końcowy ===")
print(f"x* = {res_x:.6f}, y* = {res_y:.6f}")

xs_plot = np.linspace(a, b, 1000)
ys_plot = np.array([f_np(xx) for xx in xs_plot])
pos_x = [p[0] for p in positions]
pos_y = [float(f_np(p[0])) for p in positions]

fig, ax = plt.subplots()
ax.set_xlim(a, b)
ymin_plot = min(np.nanmin(ys_plot), min(pos_y) if pos_y else 0)
ymax_plot = max(np.nanmax(ys_plot), max(pos_y) if pos_y else 1)
yrange = ymax_plot - ymin_plot
ax.set_ylim(ymin_plot - 0.1 * yrange, ymax_plot + 0.1 * yrange)
line_func, = ax.plot(xs_plot, ys_plot, linewidth=1)
path_line, = ax.plot([], [], linestyle='-', marker='o', markersize=4, color='orange')
current_point, = ax.plot([], [], marker='o', markersize=8, color='green')
title = ax.set_title('')

def init():
    path_line.set_data([], [])
    current_point.set_data([], [])
    title.set_text('')
    return path_line, current_point, title

def update(i):
    idx = min(i, len(pos_x) - 1)
    path_line.set_data(pos_x[:idx+1], pos_y[:idx+1])
    current_point.set_data([pos_x[idx]], [pos_y[idx]])
    title.set_text(f"Iteracja {idx+1}/{len(pos_x)}  x={pos_x[idx]:.6f} y={pos_y[idx]:.6f}")
    return path_line, current_point, title

ani = animation.FuncAnimation(
    fig, update, frames=len(pos_x),
    init_func=init, interval=30, blit=False, repeat=False
)
plt.show()
