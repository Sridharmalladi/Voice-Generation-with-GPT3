import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Here we define the Generator model for the GAN
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(100, 256),  
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),  
            nn.Tanh()  
        )

    def forward(self, x):
        return self.fc(x).view(-1, 1, 28, 28)  

# Define the Discriminator model for the GAN
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Sigmoid to produce probability (real/fake)
        )

    def forward(self, x):
        return self.fc(x.view(-1, 28 * 28))  # Flatten image input

# Training loop for the GAN
def train_gan(generator, discriminator, dataloader, num_epochs=100):
    # Loss function (binary cross entropy for real/fake classification)
    criterion = nn.BCELoss()

    # Optimizers for both generator and discriminator
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

    for epoch in range(num_epochs):
        for batch_idx, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)

            # Training the discriminator
            real_labels = torch.ones(batch_size, 1)  
            fake_labels = torch.zeros(batch_size, 1)  
            
            # Train on real images
            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)

            # Generate fake images
            noise = torch.randn(batch_size, 100)
            fake_images = generator(noise)
            outputs = discriminator(fake_images)
            d_loss_fake = criterion(outputs, fake_labels)

            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake

            # Backprop and update discriminator weights
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # Training the generator
            noise = torch.randn(batch_size, 100)
            fake_images = generator(noise)
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)  # Generator wants discriminator to output 1

            # Backprop and update generator weights
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss D: {d_loss.item()}, Loss G: {g_loss.item()}')

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize the Generator and Discriminator
    generator = Generator()
    discriminator = Discriminator()

    # Train the GAN
    train_gan(generator, discriminator, dataloader)
