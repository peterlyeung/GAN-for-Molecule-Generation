import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from rdkit import Chem
from rdkit.Chem import MolToSmiles, Draw
from rdkit.Chem.rdmolops import RDKFingerprint

# Define a simple generator and discriminator
class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Generate random data for training (you should replace this with your dataset)
data_dim = 100  # Dimension of random input
data_size = 1000  # Number of data points
generator_input_dim = 50  # Dimension of generator input
real_data = torch.randn(data_size, data_dim)
generator_input = torch.randn(data_size, generator_input_dim)

# Instantiate GAN components
generator = Generator(generator_input_dim, data_dim)
discriminator = Discriminator(data_dim)

# Define loss functions and optimizers
criterion = nn.BCELoss()
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# Training loop
num_epochs = 2000
batch_size = 64

for epoch in range(num_epochs):
    for i in range(0, data_size, batch_size):
        real_data_batch = real_data[i:i + batch_size]
        generator_input_batch = generator_input[i:i + batch_size]

        # Train the discriminator
        d_optimizer.zero_grad()
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        real_outputs = discriminator(real_data_batch)
        d_real_loss = criterion(real_outputs, real_labels)

        fake_data = generator(generator_input_batch)
        fake_outputs = discriminator(fake_data)
        d_fake_loss = criterion(fake_outputs, fake_labels)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()

        # Train the generator
        g_optimizer.zero_grad()
        fake_outputs = discriminator(fake_data)
        g_loss = criterion(fake_outputs, real_labels)
        g_loss.backward()
        g_optimizer.step()

    # Print losses and generate sample molecules
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

        # Generate sample molecules
        with torch.no_grad():
            num_samples = 5
            random_samples = torch.randn(num_samples, generator_input_dim)
            generated_samples = generator(random_samples)
            for sample in generated_samples:
                molecule = MolToSmiles(Chem.MolFromSmiles(sample.tolist()[0]))
                print(f"Generated Molecule: {molecule}")

# You can save the trained generator for future use
torch.save(generator.state_dict(), 'generator.pth')
