import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Seed randomness for reproducibility
np.random.seed(42)

# Step 0: NumPy Layer Implementations
# This section contains the from-scratch implementation of the neural network layers.

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # Helper function to get indices for im2col.
    # This is an efficient way to implement convolutions.
    N, C, H, W = x_shape
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    # An implementation of im2col based on some fancy indexing
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    # An implementation of col2im based on fancy indexing and np.add.at
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

class SiLU:
    def forward(self, x):
        sigmoid_x = 1 / (1 + np.exp(-x))
        self.cache = x, sigmoid_x
        return x * sigmoid_x

    def backward(self, dout):
        x, sigmoid_x = self.cache
        dx = sigmoid_x * (1 + x * (1 - sigmoid_x))
        return dout * dx

class Linear:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros(output_dim)

    def forward(self, x):
        self.cache = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        x = self.cache
        self.dW = np.dot(x.T, dout)
        self.db = np.sum(dout, axis=0)
        return np.dot(dout, self.W.T)

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.b = np.zeros(out_channels)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        N, C, H, W = x.shape
        F, _, HH, WW = self.W.shape
        x_col = im2col_indices(x, HH, WW, self.padding, self.stride)
        W_col = self.W.reshape(F, -1)
        out = np.dot(W_col, x_col) + self.b.reshape(-1, 1)
        out_H = (H + 2 * self.padding - HH) // self.stride + 1
        out_W = (W + 2 * self.padding - WW) // self.stride + 1
        out = out.reshape(F, out_H, out_W, N).transpose(3, 0, 1, 2)
        self.cache = x, x_col, W_col
        return out

    def backward(self, dout):
        x, x_col, W_col = self.cache
        N, C, H, W = x.shape
        F, _, HH, WW = self.W.shape
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(F, -1)
        self.db = np.sum(dout, axis=(0, 2, 3))
        self.dW = np.dot(dout_reshaped, x_col.T).reshape(self.W.shape)
        dx_col = np.dot(W_col.T, dout_reshaped)
        dx = col2im_indices(dx_col, x.shape, HH, WW, self.padding, self.stride)
        return dx

class ConvTranspose2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=1):
        # Weights are stored as (in_channels, out_channels, HH, WW)
        self.W = np.random.randn(in_channels, out_channels, kernel_size, kernel_size) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.b = np.zeros(out_channels)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        self.cache = x
        N, C_in, H_in, W_in = x.shape
        _, C_out, HH, WW = self.W.shape

        H_out = (H_in - 1) * self.stride - 2 * self.padding + HH
        W_out = (W_in - 1) * self.stride - 2 * self.padding + WW
        out_shape = (N, C_out, H_out, W_out)

        # This is equivalent to the dx calculation in Conv2D.backward
        x_reshaped = x.transpose(1, 2, 3, 0).reshape(C_in, -1)
        W_reshaped = self.W.reshape(C_in, -1)
        dx_col = np.dot(W_reshaped.T, x_reshaped)

        y = col2im_indices(dx_col, out_shape, HH, WW, self.padding, self.stride)

        return y + self.b[None, :, None, None]

    def backward(self, dout):
        x = self.cache
        N, C_in, H_in, W_in = x.shape
        _, C_out, HH, WW = self.W.shape

        self.db = np.sum(dout, axis=(0, 2, 3))

        # dx (gradient w.r.t. input) is a standard convolution of dout with W
        W_conv = self.W.transpose(1, 0, 2, 3)
        W_conv_col = W_conv.reshape(C_in, -1)
        dout_col = im2col_indices(dout, HH, WW, self.padding, self.stride)
        dx_col = np.dot(W_conv_col, dout_col)
        dx = dx_col.reshape(C_in, H_in, W_in, N).transpose(3, 0, 1, 2)

        # dW (gradient w.r.t. weights) is a convolution of x with dout
        # We need to be careful with the shapes and transpose operations
        x_reshaped = x.transpose(1, 0, 2, 3) # (C_in, N, H_in, W_in)
        dout_reshaped = dout.transpose(1, 0, 2, 3) # (C_out, N, H_out, W_out)

        dW = np.zeros_like(self.W)
        for i in range(C_in):
            for j in range(C_out):
                # Convolve each input channel with each output gradient channel
                x_slice = x_reshaped[i, :, :, :]
                dout_slice = dout_reshaped[j, :, :, :]
                # Use Conv2D logic for dW: dot(dout_col, x_col.T)
                dout_slice_col = im2col_indices(dout_slice[:,None,:,:], H_in, W_in, padding=self.padding, stride=self.stride)
                x_slice_reshaped = x_slice.reshape(N, -1)
                dW[i, j] = np.dot(x_slice_reshaped, dout_slice_col.T).reshape(self.W.shape[2:])

        return dx

class GroupNorm:
    def __init__(self, num_groups, num_channels, eps=1e-5):
        self.gamma = np.ones(num_channels)
        self.beta = np.zeros(num_channels)
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.shape
        x_reshaped = x.reshape(N, self.num_groups, C // self.num_groups, H, W)
        mean = np.mean(x_reshaped, axis=(2, 3, 4), keepdims=True)
        var = np.var(x_reshaped, axis=(2, 3, 4), keepdims=True)
        x_norm = (x_reshaped - mean) / np.sqrt(var + self.eps)
        x_norm = x_norm.reshape(N, C, H, W)
        out = self.gamma.reshape(1, -1, 1, 1) * x_norm + self.beta.reshape(1, -1, 1, 1)
        self.cache = x, x_norm, mean, var, x_reshaped.shape
        return out

    def backward(self, dout):
        x, x_norm, mean, var, x_reshaped_shape = self.cache
        N, C, H, W = x.shape
        self.dgamma = np.sum(dout * x_norm, axis=(0, 2, 3))
        self.dbeta = np.sum(dout, axis=(0, 2, 3))
        dx_norm = dout * self.gamma.reshape(1, -1, 1, 1)
        dx_norm_reshaped = dx_norm.reshape(x_reshaped_shape)
        x_reshaped = x.reshape(x_reshaped_shape)
        dvar = np.sum(dx_norm_reshaped * (x_reshaped - mean) * -0.5 * (var + self.eps)**(-1.5), axis=(2, 3, 4), keepdims=True)
        dmean = np.sum(dx_norm_reshaped * -1 / np.sqrt(var + self.eps), axis=(2, 3, 4), keepdims=True)
        M = x_reshaped.shape[2] * x_reshaped.shape[3] * x_reshaped.shape[4]
        dx = dx_norm_reshaped / np.sqrt(var + self.eps) + \
             dvar * 2 * (x_reshaped - mean) / M + \
             dmean / M
        return dx.reshape(N, C, H, W)

class NumPyUNet:
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=16, text_emb_dim=16):
        self.layers = {}
        self.layers['time_mlp_l1'] = Linear(time_emb_dim, 64)
        self.layers['time_mlp_a1'] = SiLU()
        self.layers['time_mlp_l2'] = Linear(64, 64)
        self.layers['text_mlp_l1'] = Linear(text_emb_dim, 64)
        self.layers['text_mlp_a1'] = SiLU()
        self.layers['text_mlp_l2'] = Linear(64, 64)
        self.layers['enc_conv1'] = Conv2D(in_channels, 32, 3)
        self.layers['enc_gn1'] = GroupNorm(8, 32)
        self.layers['enc_act1'] = SiLU()
        self.layers['enc_down1'] = Conv2D(32, 32, 4, stride=2)
        self.layers['bot_conv2'] = Conv2D(32, 64, 3)
        self.layers['bot_gn2'] = GroupNorm(16, 64)
        self.layers['bot_act2'] = SiLU()
        self.layers['dec_up1'] = ConvTranspose2D(64, 64, 4, stride=2)
        self.layers['dec_conv3'] = Conv2D(96, 32, 3)
        self.layers['dec_gn3'] = GroupNorm(8, 32)
        self.layers['dec_act3'] = SiLU()
        self.layers['dec_out_conv'] = Conv2D(32, out_channels, 3)

    def forward(self, x, t_emb, c_emb):
        time_proj = self.layers['time_mlp_l1'].forward(t_emb)
        time_proj = self.layers['time_mlp_a1'].forward(time_proj)
        time_proj = self.layers['time_mlp_l2'].forward(time_proj)
        text_proj = self.layers['text_mlp_l1'].forward(c_emb)
        text_proj = self.layers['text_mlp_a1'].forward(text_proj)
        text_proj = self.layers['text_mlp_l2'].forward(text_proj)
        emb_proj = time_proj + text_proj
        h1 = self.layers['enc_conv1'].forward(x)
        h1 = self.layers['enc_gn1'].forward(h1)
        h1 = self.layers['enc_act1'].forward(h1)
        h_down = self.layers['enc_down1'].forward(h1)
        h2 = self.layers['bot_conv2'].forward(h_down)
        h2 = self.layers['bot_gn2'].forward(h2)
        h2 = self.layers['bot_act2'].forward(h2)
        emb_proj_reshaped = emb_proj[:, :, np.newaxis, np.newaxis]
        h2 = h2 + emb_proj_reshaped
        h_up = self.layers['dec_up1'].forward(h2)
        h_skip_combined = np.concatenate([h_up, h1], axis=1)
        h3 = self.layers['dec_conv3'].forward(h_skip_combined)
        h3 = self.layers['dec_gn3'].forward(h3)
        h3 = self.layers['dec_act3'].forward(h3)
        output = self.layers['dec_out_conv'].forward(h3)
        return output

    def backward(self, dout):
        d_h3 = self.layers['dec_out_conv'].backward(dout)
        d_h3 = self.layers['dec_act3'].backward(d_h3)
        d_h3 = self.layers['dec_gn3'].backward(d_h3)
        d_skip_combined = self.layers['dec_conv3'].backward(d_h3)
        d_h_up = d_skip_combined[:, :64, :, :]
        d_h1_skip = d_skip_combined[:, 64:, :, :]
        d_h2 = self.layers['dec_up1'].backward(d_h_up)
        d_emb_proj_reshaped = d_h2
        self.d_emb_proj = np.sum(d_emb_proj_reshaped, axis=(2, 3))
        d_h2 = d_h2
        d_h2 = self.layers['bot_act2'].backward(d_h2)
        d_h2 = self.layers['bot_gn2'].backward(d_h2)
        d_h_down = self.layers['bot_conv2'].backward(d_h2)
        d_h1_down = self.layers['enc_down1'].backward(d_h_down)
        d_h1 = d_h1_down + d_h1_skip
        d_h1 = self.layers['enc_act1'].backward(d_h1)
        d_h1 = self.layers['enc_gn1'].backward(d_h1)
        self.layers['enc_conv1'].backward(d_h1)
        d_time_proj = self.d_emb_proj
        d_text_proj = self.d_emb_proj
        d_time_proj = self.layers['time_mlp_l2'].backward(d_time_proj)
        d_time_proj = self.layers['time_mlp_a1'].backward(d_time_proj)
        self.layers['time_mlp_l1'].backward(d_time_proj)
        d_text_proj = self.layers['text_mlp_l2'].backward(d_text_proj)
        d_text_proj = self.layers['text_mlp_a1'].backward(d_text_proj)
        self.layers['text_mlp_l1'].backward(d_text_proj)

    def get_params(self):
        params = {}
        for name, layer in self.layers.items():
            if hasattr(layer, 'W'): params[f'{name}_W'] = layer.W
            if hasattr(layer, 'b'): params[f'{name}_b'] = layer.b
            if hasattr(layer, 'gamma'): params[f'{name}_gamma'] = layer.gamma
            if hasattr(layer, 'beta'): params[f'{name}_beta'] = layer.beta
        return params

    def set_params(self, params):
        for name, layer in self.layers.items():
            if f'{name}_W' in params: layer.W = params[f'{name}_W']
            if f'{name}_b' in params: layer.b = params[f'{name}_b']
            if f'{name}_gamma' in params: layer.gamma = params[f'{name}_gamma']
            if f'{name}_beta' in params: layer.beta = params[f'{name}_beta']

# Where to persist the trained model so the script works regardless of CWD
MODEL_PATH = Path(__file__).resolve().parent / "trained_numpy_unet_model.pkl"

# Step 1: Define The Training Data
print("\nStep 1: Defining The Training Data")
training_data = [
    {"prompt": "a happy face", "matrix": np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1,-1,-1,-1,-1,-1,-1, 1],
    [1,-1, 1,-1,-1, 1,-1, 1],
    [1,-1,-1,-1,-1,-1,-1, 1],
    [1,-1, 1,-1,-1, 1,-1, 1],
    [1,-1, 1, 1, 1, 1,-1, 1],
    [1,-1,-1,-1,-1,-1,-1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
    ],dtype=np.float32)},
    {"prompt": "a sad face", "matrix": np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1,-1,-1,-1,-1,-1,-1, 1],
    [1,-1, 1,-1,-1, 1,-1, 1],
    [1,-1,-1,-1,-1,-1,-1, 1],
    [1,-1, 1, 1, 1, 1,-1, 1],
    [1,-1, 1,-1,-1, 1,-1, 1],
    [1,-1,-1,-1,-1,-1,-1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
    ],dtype=np.float32)},
    {"prompt": "a meh face", "matrix": np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1,-1,-1,-1,-1,-1,-1, 1],
    [1,-1, 1,-1,-1, 1,-1, 1],
    [1,-1,-1,-1,-1,-1,-1, 1],
    [1,-1,-1,-1,-1,-1,-1, 1],
    [1,-1, 1, 1, 1, 1,-1, 1],
    [1,-1,-1,-1,-1,-1,-1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
    ],dtype=np.float32)},
]
text_embeddings = {
    "a happy face": np.array([1.0, 0.5, 0.25, 0.125] * 4),
    "a sad face":   np.array([0.125, 0.25, 0.5, 1.] * 4),
    "a meh face":   np.array([1., 1., 0., 0.] * 4)
}

# --- Visualize the training data ---
print("\nVisualizing The Training Data")
# Create a figure to hold the images
# The `figsize` is adjusted for better viewing of 3 images.
fig, axes = plt.subplots(1, len(training_data), figsize=(12, 4))
fig.suptitle("Initial Training Data Images", fontsize=16)

for i, data_point in enumerate(training_data):
    matrix = data_point["matrix"]
    prompt = data_point["prompt"]

    # Use imshow to display the matrix. 'gray' colormap is good for this kind of data.
    axes[i].imshow(matrix, cmap='gray_r')
    axes[i].set_title(prompt)
    # Turn off the axis numbers for a cleaner look
    axes[i].axis('off')

plt.show()

# Step 2: Define The Forward Diffusion & Positional Encoding
num_timesteps = 1000
beta_start = 0.0001
beta_end = 0.02
betas = np.linspace(beta_start, beta_end, num_timesteps)
alphas = 1. - betas
alphas_cumprod = np.cumprod(alphas)

def add_noise(original_image, t_idx):
    sqrt_alpha_cumprod = np.sqrt(alphas_cumprod[t_idx])
    sqrt_one_minus_alpha_cumprod = np.sqrt(1. - alphas_cumprod[t_idx])
    noise = np.random.randn(*original_image.shape)
    noised_image = sqrt_alpha_cumprod * original_image + sqrt_one_minus_alpha_cumprod * noise
    return noised_image, noise

def create_timestep_embedding(t, embedding_dim):
    position = t; half_dim = embedding_dim // 2
    exponent = -math.log(10000.0) * np.arange(half_dim) / half_dim
    embedding_base = np.exp(exponent); embedding = position * embedding_base
    return np.concatenate([np.sin(embedding), np.cos(embedding)])

print("\nStep 2: Visualizing The Forward Diffusion Process")

# Define the timesteps you want to visualize
steps_to_show = [0, 100, 300, 600, 750, 999]
fig, axes = plt.subplots(1, len(steps_to_show), figsize=(18, 3))
fig.suptitle("Forward Diffusion: Image to Noise", fontsize=16)

# Generate and plot the noised images
for i, t in enumerate(steps_to_show):
    noisy_image, _ = add_noise(training_data[0]["matrix"], t)  # Use the first training image for demonstration

    axes[i].imshow(noisy_image, cmap='gray_r', vmin=-1.5, vmax=1.5)
    axes[i].set_title(f"Timestep t={t+1}")
    axes[i].axis('off')

    # print noise data for debugging to 3 decimal places
    noise_data = add_noise(training_data[0]["matrix"], t)[1]
    print(f"Noise at timestep {t+1}:\n{np.round(noise_data, 2)}\n")

plt.show()

# Render 1st noise image and accompanying matrix of numbers next to it
print("\nExample of Noise Image and Matrix")
example_timestep = 1000
example_noisy_image, example_actual_noise = add_noise(training_data[0]["matrix"], example_timestep - 1)
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.imshow(example_noisy_image, cmap='gray_r', vmin=-1.5, vmax=1.5)
plt.title(f"Noisy Image at t={example_timestep}")
plt.axis('off')
# Show matrix as numbers rounded to 2 decimal places
plt.subplot(1, 2, 2)
plt.title("Image Matrix")
plt.axis('off')
plt.text(0.5, 0.5, np.round(example_noisy_image, 2), fontsize=8, ha='center', va='center', transform=plt.gca().transAxes)
plt.show()
print("Example Noisy Image Matrix:\n", np.round(example_noisy_image, 2))




# Step 3: The Training Loop with NumPy U-Net
print("\nStep 3: Starting Training Loop")
learning_rate = 1e-4
num_epochs = 10000
beta1_adam, beta2_adam, epsilon_adam = 0.9, 0.999, 1e-8

model = NumPyUNet()
params = model.get_params()
m, v = {k: np.zeros_like(p) for k, p in params.items()}, {k: np.zeros_like(p) for k, p in params.items()}

losses = []












# --- Illustrative Example for Projected Embeddings ---
# 2. Define the specific timestep
example_timestep = 22

# 3. Calculate and print the INITIAL 16-dim embedding
initial_time_emb = create_timestep_embedding(example_timestep, 16)
print("--- Initial 16-dim Sinusoidal Embedding for t=22 ---")
print("(This should match your manual calculation)")
print(np.round(initial_time_emb, 2))
print("\nShape:", initial_time_emb.shape)

time_mlp_l1 = Linear(16, 64)
time_mlp_a1 = SiLU()
time_mlp_l2 = Linear(64, 64)
# 4. Pass the initial embedding through the MLP
# Add a batch dimension of 1
initial_time_emb_batch = initial_time_emb[np.newaxis, :]

# Forward pass through the MLP
time_proj = time_mlp_l1.forward(initial_time_emb_batch)
time_proj = time_mlp_a1.forward(time_proj)
final_time_proj = time_mlp_l2.forward(time_proj)

# 5. Print the FINAL 64-dim projected embedding to 2 decimal places
print("\n\n--- Final 64-dim Projected Embedding for t=22 (after MLP) ---")
print("(This is the result of transforming the vector above)")
print(np.round(final_time_proj, 2))
print("\nShape:", final_time_proj.shape)




# 1. Define the specific input
example_prompt = "a happy face"

# 2. Get the initial 16-dim embedding
initial_text_emb = text_embeddings[example_prompt]
print("--- Initial 16-dim Text Embedding for 'a happy face' ---")
print(np.round(initial_text_emb, 3))
print("\nShape:", initial_text_emb.shape)

# 3. Define the MLP for the text embedding
text_mlp_l1 = Linear(16, 64)
text_mlp_a1 = SiLU()
text_mlp_l2 = Linear(64, 64)

# 4. Pass the initial embedding through the MLP
# Add a batch dimension of 1
initial_text_emb_batch = initial_text_emb[np.newaxis, :]

# Forward pass through the MLP
text_proj = text_mlp_l1.forward(initial_text_emb_batch)
text_proj = text_mlp_a1.forward(text_proj)
final_text_proj = text_mlp_l2.forward(text_proj)

# 5. Print the FINAL 64-dim projected embedding to 2 decimal places
print("\n\n--- Final 64-dim Projected Embedding for 'a happy face' (after MLP) ---")
print("(This is the result of transforming the vector above)")
print(np.round(final_text_proj, 2))
print("\nShape:", final_text_proj.shape)
















print(f"Training NumPy U-Net model for {num_epochs} epochs.")

# Show math example for noise image + position + text embedding
print("\n--- Example of matrix manipulation for embedding ---")
example_data = training_data[0]["matrix"]
print("Original Image Matrix:\n", example_data)
example_text_embedding = text_embeddings[training_data[0]["prompt"]]
print("Text Embedding:\n", example_text_embedding)
example_timestep = 500
example_timestep_embedding = create_timestep_embedding(example_timestep, 16)
print("Timestep Embedding:\n", example_timestep_embedding)
print("\n--- Final U-net input ---")
example_noisy_image, example_actual_noise = add_noise(example_data, example_timestep - 1)
print("Noisy Image Matrix:\n", example_noisy_image)
print("Actual Noise Matrix:\n", example_actual_noise)




for epoch in range(num_epochs):
    data_point = training_data[np.random.randint(0, len(training_data))]
    image_matrix = data_point["matrix"]
    text_embedding = text_embeddings[data_point["prompt"]]

    t = np.random.randint(1, num_timesteps)
    noisy_image, actual_noise = add_noise(image_matrix, t-1)

    noisy_image_batch = noisy_image[np.newaxis, np.newaxis, :, :]
    text_emb_batch = text_embedding[np.newaxis, :]
    timestep_emb_batch = create_timestep_embedding(t, 16)[np.newaxis, :]

    predicted_noise = model.forward(noisy_image_batch, timestep_emb_batch, text_emb_batch)

    loss = np.mean((predicted_noise - actual_noise[np.newaxis, np.newaxis, :, :])**2)
    losses.append(loss)

    d_loss = 2 * (predicted_noise - actual_noise[np.newaxis, np.newaxis, :, :]) / predicted_noise.size
    model.backward(d_loss)

    t_adam = epoch + 1
    grads = {f'{name}_W': layer.dW for name, layer in model.layers.items() if hasattr(layer, 'dW')}
    grads.update({f'{name}_b': layer.db for name, layer in model.layers.items() if hasattr(layer, 'db')})
    grads.update({f'{name}_gamma': layer.dgamma for name, layer in model.layers.items() if hasattr(layer, 'dgamma')})
    grads.update({f'{name}_beta': layer.dbeta for name, layer in model.layers.items() if hasattr(layer, 'dbeta')})

    for k in params.keys():
        if k not in grads: continue # Skip if no gradient was computed for a parameter
        m[k] = beta1_adam * m[k] + (1 - beta1_adam) * grads[k]
        v[k] = beta2_adam * v[k] + (1 - beta2_adam) * (grads[k]**2)
        m_hat = m[k] / (1 - beta1_adam**t_adam)
        v_hat = v[k] / (1 - beta2_adam**t_adam)
        params[k] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon_adam)

    model.set_params(params)

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.8f}")

print("\n--- Training Complete ---")
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title("Training Loss Over Time (NumPy U-Net)")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error Loss")
plt.yscale('log')
plt.grid(True)
plt.show()

print("\n--- Saving Model Weights ---")
with MODEL_PATH.open('wb') as f:
    pickle.dump(model.get_params(), f)
print(f"Model weights saved to '{MODEL_PATH}'")
