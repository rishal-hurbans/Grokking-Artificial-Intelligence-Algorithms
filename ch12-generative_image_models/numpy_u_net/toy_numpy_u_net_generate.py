import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Step 0: Redefine the NumPy Layers and UNet Class
# We must have the exact same class definitions as the training script
# to be able to reconstruct the model and load its parameters.

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
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
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
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
        # No need to cache for inference
        return x / (1 + np.exp(-x))

class Linear:
    def __init__(self, input_dim, output_dim):
        # Initialize with zeros, will be overwritten by loaded params
        self.W = np.zeros((input_dim, output_dim))
        self.b = np.zeros(output_dim)
    def forward(self, x):
        return np.dot(x, self.W) + self.b

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        self.W = np.zeros((out_channels, in_channels, kernel_size, kernel_size))
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
        return out.reshape(F, out_H, out_W, N).transpose(3, 0, 1, 2)

class ConvTranspose2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=1):
        self.W = np.zeros((in_channels, out_channels, kernel_size, kernel_size))
        self.b = np.zeros(out_channels)
        self.stride = stride
        self.padding = padding
    def forward(self, x):
        N, C_in, H_in, W_in = x.shape
        _, C_out, HH, WW = self.W.shape
        H_out = (H_in - 1) * self.stride - 2 * self.padding + HH
        W_out = (W_in - 1) * self.stride - 2 * self.padding + WW
        out_shape = (N, C_out, H_out, W_out)
        x_reshaped = x.transpose(1, 2, 3, 0).reshape(C_in, -1)
        W_reshaped = self.W.reshape(C_in, -1)
        dx_col = np.dot(W_reshaped.T, x_reshaped)
        y = col2im_indices(dx_col, out_shape, HH, WW, self.padding, self.stride)
        return y + self.b[None, :, None, None]

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
        return self.gamma.reshape(1, -1, 1, 1) * x_norm + self.beta.reshape(1, -1, 1, 1)

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
        return self.layers['dec_out_conv'].forward(h3)

    def set_params(self, params):
        for name, layer in self.layers.items():
            if hasattr(layer, 'W') and f'{name}_W' in params: layer.W = params[f'{name}_W']
            if hasattr(layer, 'b') and f'{name}_b' in params: layer.b = params[f'{name}_b']
            if hasattr(layer, 'gamma') and f'{name}_gamma' in params: layer.gamma = params[f'{name}_gamma']
            if hasattr(layer, 'beta') and f'{name}_beta' in params: layer.beta = params[f'{name}_beta']

# Step 1: Load Model and Define Helpers
print("\nStep 1: Loading Model and Defining Helpers")
model = NumPyUNet()
MODEL_PATH = Path(__file__).resolve().parent / "trained_numpy_unet_model.pkl"
try:
    with MODEL_PATH.open('rb') as f:
        params = pickle.load(f)
    model.set_params(params)
    print("Successfully loaded trained NumPy U-Net model weights.")
except FileNotFoundError:
    print(f"Error: '{MODEL_PATH}' not found.")
    print("Please run the NumPy training script first.")
    exit()

num_timesteps = 50
beta_start = 0.0001
beta_end = 0.02
betas = np.linspace(beta_start, beta_end, num_timesteps)
alphas = 1. - betas
alphas_cumprod = np.cumprod(alphas)

def create_timestep_embedding(t, embedding_dim):
    position = t; half_dim = embedding_dim // 2
    exponent = -math.log(10000.0) * np.arange(half_dim) / half_dim
    embedding_base = np.exp(exponent); embedding = position * embedding_base
    return np.concatenate([np.sin(embedding), np.cos(embedding)])

text_embeddings = {
    "a happy face": np.array([1., 0.5, 0.25, 0.125] * 4),
    "a sad face":   np.array([0.125, 0.25, 0.5, 1.] * 4),
    "a meh face":   np.array([1., 1., 0., 0.] * 4)
}

# Choose a prompt to generate
prompt_to_generate = "a happy face"
text_embedding = text_embeddings[prompt_to_generate]

# Step 2: Generating an Image (Inference)
print(f"\nStarting Image Generation for prompt: '{prompt_to_generate}'")
np.random.seed(42)
# Start with random noise, shaped with a batch dimension
generated_image = np.random.randn(1, 1, 8, 8)
generation_steps = [np.squeeze(generated_image)]
print("1. Initial Random Noise Image created.")

for t in reversed(range(1, num_timesteps)):
    if t % 100 == 0: print(f"Denoising at timestep {t}...")

    # Prepare inputs with batch dimension
    timestep_emb = create_timestep_embedding(t, 16)[np.newaxis, :]
    text_emb = text_embedding[np.newaxis, :]

    # Predict noise using the U-Net
    predicted_noise = model.forward(generated_image, timestep_emb, text_emb)

    # Draw predicted noise for timestep 1 only
    if t == num_timesteps - 1:
        plt.figure(figsize=(4, 4))
        plt.imshow(np.clip(predicted_noise[0, 0], -1, 1), cmap='gray_r', interpolation='nearest')
        plt.title(f"Predicted Noise at Timestep {t}")
        plt.axis('off')
        plt.show()

    # Get diffusion constants for this timestep
    alpha_t = alphas[t-1]
    alpha_t_cumprod = alphas_cumprod[t-1]

    # The reverse diffusion formula
    term1 = 1 / np.sqrt(alpha_t)
    term2 = (1 - alpha_t) / np.sqrt(1 - alpha_t_cumprod)
    generated_image = term1 * (generated_image - term2 * predicted_noise)

    # Add noise back in for all but the last step
    if t > 1:
        z = np.random.randn(*generated_image.shape)
        beta_t = betas[t-1]
        generated_image += np.sqrt(beta_t) * z

    # Store a snapshot for visualization
    generation_steps.append(np.squeeze(generated_image))

# Step 3: Inference Visualization
print("\nGeneration Complete")
fig, axs = plt.subplots(1, 6, figsize=(20, 4))
# Show steps from noise to final image
steps_to_show = [1, 9, 19, 29, 39, 49]
#steps_to_show = [3, 4, 700, 800, 850, 999]
#steps_to_show = [1, 50, 100, 150, 200, 250]
for i, step_index in enumerate(steps_to_show):
    img = np.clip(generation_steps[step_index], -1, 1)
    title = f"Start (Noise)" if step_index == 0 else f"After step {num_timesteps - step_index}"
    axs[i].imshow(img, cmap='gray_r', interpolation='nearest')
    axs[i].set_title(title)
    axs[i].axis('off')
plt.suptitle(f"Reverse Diffusion for: '{prompt_to_generate}'", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Print alpha and alpha_cumprod values for 899 and 900 timesteps
print(f"\nAlpha at step 899: {alphas[0]:.6f}, Alpha_cumprod at step 900: {alphas_cumprod[0]:.6f}")
print(f"Alpha at step 900: {alphas[1]:.6f}, Alpha_cumprod at step 901: {alphas_cumprod[1]:.6f}")

# Display the final generated image separately
print("\nFinal Generated Image:")
plt.figure(figsize=(4, 4))
final_image = np.clip(generation_steps[-1], -1, 1)
plt.imshow(final_image, cmap='gray_r', interpolation='nearest')
plt.title(f"Final Result for '{prompt_to_generate}'")
plt.axis('off')
plt.show()
