import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math

# Use a GPU if available, otherwise use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Define the Training Data
print("\nStep 1: Defining The Training Data")
training_data = [
    {
        "prompt": "a happy face",
        "matrix": torch.tensor(np.array([
            [ 1,  1,  1,  1,  1,  1,  1,  1], [ 1, -1, -1, -1, -1, -1, -1,  1],
            [ 1, -1,  1, -1, -1,  1, -1,  1], [ 1, -1, -1, -1, -1, -1, -1,  1],
            [ 1, -1,  1, -1, -1,  1, -1,  1], [ 1, -1,  1,  1,  1,  1, -1,  1],
            [ 1, -1, -1, -1, -1, -1, -1,  1], [ 1,  1,  1,  1,  1,  1,  1,  1]
        ]), dtype=torch.float32)
    },
    {
        "prompt": "a sad face",
        "matrix": torch.tensor(np.array([
            [ 1,  1,  1,  1,  1,  1,  1,  1], [ 1, -1, -1, -1, -1, -1, -1,  1],
            [ 1, -1,  1, -1, -1,  1, -1,  1], [ 1, -1, -1, -1, -1, -1, -1,  1],
            [ 1, -1, -1, -1, -1, -1, -1,  1], [ 1, -1,  1,  1,  1,  1, -1,  1],
            [ 1, -1,  1, -1, -1,  1, -1,  1], [ 1,  1,  1,  1,  1,  1,  1,  1]
        ]), dtype=torch.float32)
    },
    {
        "prompt": "a meh face",
        "matrix": torch.tensor(np.array([
            [ 1,  1,  1,  1,  1,  1,  1,  1], [ 1, -1, -1, -1, -1, -1, -1,  1],
            [ 1, -1,  1, -1, -1,  1, -1,  1], [ 1, -1, -1, -1, -1, -1, -1,  1],
            [ 1, -1,  1,  1,  1,  1,  1,  1], [ 1, -1, -1, -1, -1, -1, -1,  1],
            [ 1, -1, -1, -1, -1, -1, -1,  1], [ 1,  1,  1,  1,  1,  1,  1,  1]
        ]), dtype=torch.float32)
    }
]
# Text embeddings also become tensors
text_embeddings = {
    "a happy face": torch.tensor(np.array([1., 0.5, 0.25, 0.125] * 4), dtype=torch.float32),
    "a sad face":   torch.tensor(np.array([0.125, 0.25, 0.5, 1.] * 4), dtype=torch.float32),
    "a meh face":   torch.tensor(np.array([1., 1., 0., 0.] * 4), dtype=torch.float32)
}
print(f"Defined {len(training_data)} training examples.")


# Step 2: Define Forward Diffusion & Positional Encoding
print("\nStep 2: Defining The Forward Diffusion & Positional Encoding")
num_timesteps = 1000
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# Helper function to get the alpha values for a specific timestep
def get_alpha_values(t):
    # t is an integer tensor, so we need to subtract 1 for 0-based indexing
    return alphas_cumprod.gather(-1, t-1)

def add_noise(original_image, t):
    sqrt_alpha_cumprod = torch.sqrt(get_alpha_values(t))
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1. - get_alpha_values(t))
    noise = torch.randn_like(original_image)

    # Reshape alphas to allow broadcasting with the image
    sqrt_alpha_cumprod = sqrt_alpha_cumprod.view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.view(-1, 1, 1, 1)

    noised_image = sqrt_alpha_cumprod * original_image + sqrt_one_minus_alpha_cumprod * noise
    return noised_image, noise

def create_timestep_embedding(t, embedding_dim):
    half_dim = embedding_dim // 2
    exponent = -math.log(10000.0) * torch.arange(half_dim, device=device) / half_dim
    embedding_base = torch.exp(exponent)
    embedding = t.float().to(device) * embedding_base
    return torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=-1)


# Step 3: Define The U-Net Architecture
print("\nStep 3: Defining The U-Net Architecture")

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=16, text_emb_dim=16):
        super().__init__()

        # Time and Text Embedding Projectors
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        self.text_mlp = nn.Sequential(
            nn.Linear(text_emb_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )

        # Encoder (Contracting Path)
        # Input: 8x8
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)
        self.act1 = nn.SiLU()
        self.downsample1 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1) # 8x8 -> 4x4

        # Bottleneck
        # Input: 4x4
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(16, 64)
        self.act2 = nn.SiLU()

        # Decoder (Expansive Path)
        self.upsample1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1) # 4x4 -> 8x8

        # The input channels for conv3 is 64 (from upsample) + 32 (from skip connection) = 96
        self.conv3 = nn.Conv2d(96, 32, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(8, 32)
        self.act3 = nn.SiLU()

        # Final Output Layer
        self.output_conv = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t_emb, c_emb):
        # Project Time and Text embeddings
        time_proj = self.time_mlp(t_emb)
        text_proj = self.text_mlp(c_emb)
        emb_proj = time_proj + text_proj # Combine embeddings

        # Encoder
        # x is the input image (batch, 1, 8, 8)
        h1 = self.act1(self.gn1(self.conv1(x)))
        h_down = self.downsample1(h1)

        # Bottleneck
        h2 = self.act2(self.gn2(self.conv2(h_down)))

        # Inject the combined embeddings into the bottleneck features
        # Reshape emb_proj to (batch, channels, 1, 1) to allow broadcasting
        emb_proj_reshaped = emb_proj[:, :, None, None]
        h2 = h2 + emb_proj_reshaped

        # Decoder
        h_up = self.upsample1(h2)

        # Concatenate the skip connection from the encoder with the upsampled features
        # This is the defining feature of a U-Net!
        skip_connection = h1
        h_skip_combined = torch.cat([h_up, skip_connection], dim=1) # dim=1 is the channel dimension

        h3 = self.act3(self.gn3(self.conv3(h_skip_combined)))

        # Final Output
        output = self.output_conv(h3)
        return output


# Step 4: The Training Loop
print("\nStep 4: Starting Training Loop")
learning_rate = 1e-3
num_epochs = 5000 # U-Nets are efficient, often need fewer epochs

# Initialize the model, loss function, and optimizer
model = UNet().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses = []
print(f"Training U-Net model for {num_epochs} epochs.")

for epoch in range(num_epochs):
    # Data Preparation
    data_point = training_data[np.random.randint(0, len(training_data))]
    image_matrix = data_point["matrix"].to(device)
    # Use a different variable name to avoid shadowing the dictionary
    current_text_embedding_1d = text_embeddings[data_point["prompt"]].to(device)

    # Add a batch and channel dimension: (8, 8) -> (1, 1, 8, 8)
    image_matrix = image_matrix.unsqueeze(0).unsqueeze(0)

    # Create a random timestep t for this image
    t_int = torch.randint(1, num_timesteps + 1, (1,), device=device)

    # Forward Diffusion
    noisy_image, actual_noise = add_noise(image_matrix, t_int)

    # Prepare Embeddings for the Model
    timestep_embedding_1d = create_timestep_embedding(t_int, 16)

    # Add a batch dimension to the embeddings to match the image batch
    # Shape changes from [16] to [1, 16]
    timestep_embedding = timestep_embedding_1d.unsqueeze(0)
    text_embedding = current_text_embedding_1d.unsqueeze(0)

    # Model Prediction
    # PyTorch automatically handles the forward pass and tracks gradients
    predicted_noise = model(noisy_image, timestep_embedding, text_embedding)

    # Loss Calculation
    loss = loss_fn(predicted_noise, actual_noise)
    losses.append(loss.item())

    # Backpropagation & Optimization
    optimizer.zero_grad() # Reset gradients from previous step
    loss.backward() # Compute gradients for all model parameters
    optimizer.step() # Update model weights

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.8f}")

# Training Visualization
print("\nTraining Complete")
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title("Training Loss Over Time (U-Net)")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error Loss")
plt.yscale('log')
plt.grid(True)
plt.show()

# Save the Trained Model Weights
print("\nSaving Model Weights")
torch.save(model.state_dict(), '/trained_pytorch_unet_model.pth')
print("Model weights saved to 'trained_unet_model.pth'")
