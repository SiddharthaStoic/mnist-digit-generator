import torch
from generator_model import load_generator_model

def generate_digit_images(digit, num_samples=5, z_dim=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = load_generator_model(device=device)

    noise = torch.randn(num_samples, z_dim).to(device)
    labels = torch.full((num_samples,), digit, dtype=torch.long).to(device)

    with torch.no_grad():
        generated = generator(noise, labels).cpu()

    images = []
    for img in generated:
        img = img.squeeze().numpy()
        img = (img + 1) / 2.0  # Normalize from [-1,1] to [0,1]
        images.append(img)
    return images