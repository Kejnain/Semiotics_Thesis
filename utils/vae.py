import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim=28, image_size=(200, 200)):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        
        self.flattened_size = self.calculateFlattenedSize(image_size)
        
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc21 = nn.Linear(512, self.latent_dim)
        self.fc22 = nn.Linear(512, self.latent_dim)

        self.fc3 = nn.Linear(self.latent_dim, 512)
        self.fc4 = nn.Linear(512, self.flattened_size)
        
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

    def calculateFlattenedSize(self, shape):
        x = torch.zeros(1, 3, *shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return int(torch.prod(torch.tensor(x.size())))

    def encoder(self, x, sentiment_vector):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        h1 = nn.functional.relu(self.fc1(x))
        mu = self.fc21(h1) + sentiment_vector 
        logvar = self.fc22(h1)
        return mu, logvar

    def decoder(self, z):
        h3 = nn.functional.relu(self.fc3(z))
        h4 = nn.functional.relu(self.fc4(h3))
        batch_size = z.size(0)
        h4 = h4.view(batch_size, 128, 25, 25)

        x = nn.functional.relu(self.deconv1(h4))
        x = nn.functional.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))  
        return x

    def forward(self, x, sentiment_vector):
        mu, logvar = self.encoder(x, sentiment_vector)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std