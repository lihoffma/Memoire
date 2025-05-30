import numpy as np

def entropy(p):
    if p == 0 or p == 1:
        return 0  # L'entropie est nulle si la variable est certaine
    return - (p * np.log(p) + (1 - p) * np.log(1 - p))

# Probabilités des deux issues
p1 = 0.5
p2 = 0.1

# Calcul de l'entropie en log naturel
H = entropy(p1)
print(f"Entropie en log naturel : {H:.4f}")

import numpy as np

def entropy(p, base=np.e):
    if p == 0 or p == 1:
        return 0  # L'entropie est nulle si la variable est certaine
    return - (p * np.log(p) / np.log(base) + (1 - p) * np.log(1 - p) / np.log(base))

# Probabilités des deux issues
p1 = 0.5
p2 = 0.1

# Calcul de l'entropie en log base 2
H = entropy(p1, base=2)
print(f"Entropie en log base 2 : {H:.4f}")


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_selection import f_regression, mutual_info_regression

np.random.seed(0)

# Génération de X avec une distribution uniforme
X = np.random.uniform(0, 1, (10000, 3))

y1 = X[:, 0] + np.sin(6 * np.pi * X[:, 1]) + 0.1 * np.random.randn(10000)
y2 = X[:, 1] ** 2 + X[:, 2] + 0.1 * np.random.randn(10000)

y = np.vstack((y1, y2)).T

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(X[:, 0], X[:, 1], y[:, 0], cmap='viridis', edgecolor='none')
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$y_1$")
ax.set_title("Surface plot of $y_1$")

plt.savefig("surface_plot.png", dpi=300)
plt.show()
