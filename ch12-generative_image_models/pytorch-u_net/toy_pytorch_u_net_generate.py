import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Step 1: Setup and Model Definition
print("Step 1: Setting up environment and model definition")
# Use a GPU if available, otherwise use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# We need to define the UNet class again, exactly as in the training script,
# so that PyTorch knows how to load the saved weights into the model structure.
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=16, text_emb_dim=16):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, 64), nn.SiLU(), nn.Linear(64, 64)
        )
        self.text_mlp = nn.Sequential(
            nn.Linear(text_emb_dim, 64), nn.SiLU(), nn.Linear(64, 64)
        )
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)
        self.act1 = nn.SiLU()
        self.downsample1 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(16, 64)
        self.act2 = nn.SiLU()
        self.upsample1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(96, 32, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(8, 32)
        self.act3 = nn.SiLU()
        self.output_conv = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t_emb, c_emb):
        time_proj = self.time_mlp(t_emb)
        text_proj = self.text_mlp(c_emb)
        emb_proj = time_proj + text_proj
        h1 = self.act1(self.gn1(self.conv1(x)))
        h_down = self.downsample1(h1)
        h2 = self.act2(self.gn2(self.conv2(h_down)))
        emb_proj_reshaped = emb_proj[:, :, None, None]
        h2 = h2 + emb_proj_reshaped
        h_up = self.upsample1(h2)
        skip_connection = h1
        h_skip_combined = torch.cat([h_up, skip_connection], dim=1)
        h3 = self.act3(self.gn3(self.conv3(h_skip_combined)))
        output = self.output_conv(h3)
        return output

# Step 2: Load Model and Define Helpers
print("\nStep 2: Loading Model and Defining Helpers")
# Initialize the model and load the saved weights
model = UNet().to(device)
MODEL_PATH = Path(__file__).resolve().parent / "trained_pytorch_unet_model.pth"
try:
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
    print("Successfully loaded trained U-Net model weights.")
except FileNotFoundError:
    print(f"Error: '{MODEL_PATH}' not found.")
    print("Please run the training script first to create the model file.")
    exit()
model.eval() # Set the model to evaluation mode

# Define the same helper functions and constants used during training
num_timesteps = 1000
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

def create_timestep_embedding(t, embedding_dim):
    half_dim = embedding_dim // 2
    exponent = -math.log(10000.0) * torch.arange(half_dim, device=device) / half_dim
    embedding_base = torch.exp(exponent)
    embedding = t.float().to(device) * embedding_base
    return torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=-1)

text_embeddings = {
    "a happy face": torch.tensor(np.array([1., 0.5, 0.25, 0.125] * 4), dtype=torch.float32),
    "a sad face":   torch.tensor(np.array([0.125, 0.25, 0.5, 1.] * 4), dtype=torch.float32),
    "a meh face":   torch.tensor(np.array([1., 1., 0., 0.] * 4), dtype=torch.float32)
}

# Choose an image prompt
prompt_to_generate = "a sad face"

text_embedding = text_embeddings[prompt_to_generate].to(device)

# Step 3: Generating an Image (Inference)
print(f"\nStarting Image Generation for prompt: '{prompt_to_generate}'")
torch.manual_seed(42)
# Start with random noise, shaped correctly for the model (batch, channels, height, width)
generated_image = torch.randn((1, 1, 8, 8), device=device)
generation_steps = [generated_image.squeeze().cpu().numpy()]
print("1. Initial Random Noise Image created.")

with torch.no_grad(): # We don't need to track gradients during inference
    for t in reversed(range(1, num_timesteps)):
        if t % 100 == 0:
            print(f"Denoising at timestep {t}...")

        # Prepare Inputs for the U-Net
        t_int = torch.tensor([t], device=device)
        timestep_emb = create_timestep_embedding(t_int, 16).unsqueeze(0)
        text_emb = text_embedding.unsqueeze(0)

        # Inference Forward Pass
        predicted_noise = model(generated_image, timestep_emb, text_emb)

        # Reverse Diffusion Step
        alpha_t = alphas[t-1]
        alpha_t_cumprod = alphas_cumprod[t-1]

        term1 = 1 / torch.sqrt(alpha_t)
        term2 = (1 - alpha_t) / torch.sqrt(1 - alpha_t_cumprod)
        generated_image = term1 * (generated_image - term2 * predicted_noise)

        if t > 1:
            z = torch.randn_like(generated_image)
            beta_t = betas[t-1]
            generated_image = generated_image + torch.sqrt(beta_t) * z

        # Store a copy of the image for visualization
        generation_steps.append(generated_image.squeeze().cpu().numpy())

# Step 4: Inference Visualization
print("\nGeneration Complete")
fig, axs = plt.subplots(1, 6, figsize=(20, 4))
steps_to_show = [0, 200, 400, 600, 800, 999]
for i, step_index in enumerate(steps_to_show):
    img = np.clip(generation_steps[step_index], -1, 1)
    title = f"Start (Noise)" if step_index == 0 else f"After step {step_index}"
    axs[i].imshow(img, cmap='gray_r', interpolation='nearest')
    axs[i].set_title(title)
    axs[i].axis('off')
plt.suptitle(f"Reverse Diffusion for: '{prompt_to_generate}'", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Display the final generated image separately
print("\nFinal Generated Image:")
plt.figure(figsize=(4, 4))
final_image = np.clip(generation_steps[-1], -1, 1)
plt.imshow(final_image, cmap='gray_r', interpolation='nearest')
plt.title(f"Final Result for '{prompt_to_generate}'")
plt.axis('off')
plt.show()
