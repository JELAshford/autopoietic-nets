from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from tqdm import trange
import numpy as np


def AND(inputs, max_points):
    return (inputs == max_points).astype(int)


def OR(inputs, max_points):
    return ((max_points > inputs) & (inputs > 0)).astype(int)


def XOR(inputs, max_points):
    return (inputs % 2).astype(int)


def NAND(inputs, max_points):
    return 1 - AND(inputs, max_points)


def NOR(inputs, max_points):
    return 1 - OR(inputs, max_points)


def XNOR(inputs, max_points):
    return 1 - XOR(inputs, max_points)


def H(S):
    """Compute entropy of the net (S)"""
    counts = np.unique(S, return_counts=True)[1]
    p = counts / np.prod(S.shape)
    return -np.sum(p * np.log(p))


def create_circular_mask(radius):
    h = w = radius * 2
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - int(w / 2)) ** 2 + (Y - int(h / 2)) ** 2)
    return (dist_from_center <= radius).astype(int)


# Params and conditions
np.random.seed(123)
N = 300  # neuron count -> N² neurons generated
N_ITER = 200  # number of iterations
GEQ_COND = True
FIX_EPSILON = True  # to have epsilon fixed
EPSILON_FIXED = 4  # if epsilon fixed
K = 2.5  # If not fixed -> used denominator -> At max epsilon -> N_ITER/k
RADIUS = 5  # radius for consideration

S = np.random.choice((0, 1), size=(N, N))  # init state
phi = np.zeros((N, N), dtype=int)  # track synchronization at each neuron/ensemble
epsilon = EPSILON_FIXED if FIX_EPSILON else int((N_ITER * (1 - H(S))) / K)

gates = [AND, OR, XOR, NAND, NOR, XNOR]
gate = np.random.choice(np.arange(len(gates)), (N, N))


# Setup distance mask
mask = create_circular_mask(RADIUS)
mask_h, mask_w = mask.shape
mask_points = np.sum(mask)
flat_mask_index = np.array(np.argwhere(np.pad(mask, (0, N - mask_w)).flat == 1).flat)
mid_index = int(RADIUS * 2) ** 2 // 2


# Setup plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
mat1 = ax1.imshow(S, cmap="gray", vmin=0, vmax=1)
mat2 = ax2.imshow(phi, cmap="hot", vmin=0, vmax=N_ITER)
ax1.set_title(f"Threshold (epsilon): {epsilon}")
ax1.axis("off")
ax2.set_title("phi")
ax2.axis("off")
cbar2 = fig.colorbar(mat2, ax=ax2)
cbar2.set_label("Synchronization Count (phi)")


def update(frame, *args):
    global S, phi, epsilon, gate, mask

    # State update
    new_state = np.zeros(S.shape)
    conv_out = convolve2d(S, mask, mode="same", boundary="wrap")
    for ind, gate_func in enumerate(gates):
        pos = np.argwhere(gate == ind).T
        new_state[*pos] = gate_func(conv_out[*pos], max_points=mask_points)

    # Update count of state stability (phi)
    phi = np.where(new_state == S, phi + 1, 0)

    # Update state masked
    S = new_state
    epsilon = EPSILON_FIXED if FIX_EPSILON else int((N_ITER * (1 - H(S))) / K)
    mask_ensemble = phi >= epsilon if GEQ_COND else phi == epsilon
    if np.any(mask_ensemble):
        # Identify locations that are loci for updates
        ensemble_idxs = np.argwhere(mask_ensemble)
        # Iterate over locations ...
        insert_indexes, insert_values = [], []
        for i, j in ensemble_idxs:
            # Extract points in "masked" radius around that point
            point_index = (j * N) + i
            index = flat_mask_index + point_index - mid_index
            index = np.minimum(np.maximum(index, 0), (N**2) - 1)
            # Store points and new value to assign
            insert_indexes.append(index)
            insert_values.append(np.repeat(S[i, j], len(index)))
        # Concatenate and shuffle the new assignments
        inds = np.concatenate(insert_indexes)
        vals = np.concatenate(insert_values)
        pos = np.arange(len(vals))
        np.random.shuffle(pos)
        # Assign the new values
        np.put(S, inds[pos], vals[pos])

    # Update figures images and title
    mat1.set_array(S)
    mat2.set_array(phi)
    ax1.set_title(
        f"Threshold (epsilon): {epsilon}; Fixed (k: {K if not FIX_EPSILON else None}):"
        f"{FIX_EPSILON}; phi >= epsilon: {GEQ_COND}, R: {RADIUS}"
    )


ani = FuncAnimation(fig, update, frames=trange(N_ITER), interval=1000)
ani.save("autopoietic_net.gif", writer="pillow", fps=10)
print("Finished!")
# plt.show()
