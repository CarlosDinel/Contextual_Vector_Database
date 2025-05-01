import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Simulatieparameters
np.random.seed(42)
n_vectors = 15       # aantal vectoren
n_iterations = 50    # aantal tijdstappen
S = 0.1              # solidness factor (hoe vatbaar voor verandering)
R = 1.0              # impact radius
epsilon = 1e-6       # om deling door 0 te voorkomen

# Initialiseer de vectoren in 2D (niet rondom 0 om nulvectoren te vermijden)
V = np.random.uniform(0.5, 10, size=(n_vectors, 2))

# Voor trajectplot bewaren we alle posities
trajectories = np.zeros((n_iterations+1, n_vectors, 2))
trajectories[0] = V.copy()

def cosine_similarity(v1, v2):
    # Cosine similarity tussen twee vectoren
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / (norm1 * norm2 + epsilon)

def softmax(x):
    # Numeriek stabiele softmax
    exp_x = np.exp(x - np.max(x))
    return exp_x / (np.sum(exp_x) + epsilon)

# Iteratieve update
for t in range(1, n_iterations + 1):
    V_new = V.copy()
    for i in range(n_vectors):
        neighbor_indices = []   # indices van buren binnen impact radius
        impact_forces = []
        directions = []
        distances = []
        # Zoek buren j != i
        for j in range(n_vectors):
            if i == j:
                continue
            diff = V[j] - V[i]
            D = np.linalg.norm(diff)
            # Alleen buren binnen de impact radius R
            if D < R:
                neighbor_indices.append(j)
                distances.append(D)
                directions.append(diff / (D + epsilon))
                # Bereken cosine similarity
                sim = cosine_similarity(V[i], V[j])
                # ImpactForce: we combineren de (cosine) similarity en afstand (streef een dalende functie)
                impact_force = sim / (D + epsilon)
                impact_forces.append(impact_force)
                
        if len(neighbor_indices) == 0:
            # Geen buren in de buurt, geen update
            continue
        
        # Bereken softmax gewichten voor de invloeden
        impact_forces = np.array(impact_forces)
        weights = softmax(S * impact_forces)
        
        # Bereken de samengevoegde verplaatsingsvector voor V_i
        displacement = np.zeros(2)
        for k in range(len(neighbor_indices)):
            D = distances[k]
            # Gaussian demping: verder weg levert een exponentieel kleinere impact op
            damping = np.exp(- (D**2) / (2 * R**2))
            displacement += weights[k] * impact_forces[k] * directions[k] * damping
        
        # Verplaatsing is gemoduleerd door solidness factor S
        V_new[i] += S * displacement
        
    V = V_new.copy()
    trajectories[t] = V.copy()

# Plot trajecten
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(8, 8))
colors = cm.viridis(np.linspace(0, 1, n_vectors))

# Plot de trajecten
for i in range(n_vectors):
    traj = trajectories[:, i, :]
    ax.plot(traj[:, 0], traj[:, 1], color=colors[i], lw=2, label=f'Vector {i+1}')
    ax.scatter(traj[0, 0], traj[0, 1], color=colors[i], marker='o', s=50)   # beginpunt
    ax.scatter(traj[-1, 0], traj[-1, 1], color=colors[i], marker='x', s=50)  # eindpunt

ax.set_title("Simulatie: Herpositionering van Vectoren onder Wederzijdse Invloed")
ax.set_xlabel("X-positie")
ax.set_ylabel("Y-positie")
ax.grid(True, color='gray', linestyle='--', alpha=0.5)
ax.legend(loc='upper right', fontsize='x-small', ncol=2)
plt.show()
