import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, channels_img, features_d, n_classes, img_size) -> None:
        super().__init__()

        self.img_size = img_size

        # Input: N x channels_img x 64 x 64
        self.disc = nn.Sequential(
            self._block(channels_img + 1, features_d, kernel_size=4, stride=2, padding=1, instance_norm=False), #32x32
            self._block(features_d, features_d*2, kernel_size=4, stride=2, padding=1), # 16x16
            self._block(features_d*2, features_d*4, kernel_size=4, stride=2, padding=1), # 8x8
            self._block(features_d*4, features_d*8, kernel_size=4, stride=2, padding=1), # 4x4
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0) # 1x1
        )

        self.embed = nn.Embedding(n_classes, img_size * img_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, instance_norm=True):
        layers = []
        bias = False if instance_norm else True
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        if instance_norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def forward(self, x, labels):
        # Get label embedding which will be appended to the input image as a new channel
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1) # N x C+1 x H (img_size) x W (img_size)
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g, n_classes, embed_dim) -> None:
        super().__init__()

        # Input: N x z_dim x 1 x 1
        self.gen = nn.Sequential(
            self._block(z_dim + embed_dim, features_g*16, 4, 1, 0), # Nxf_g*16x4x4
            self._block(features_g*16, features_g*8, 4, 2, 1), # 8x8
            self._block(features_g*8, features_g*4, 4, 2, 1), # 16x16
            self._block(features_g*4, features_g*2, 4, 2, 1), # 32x32
            self._block(features_g*2, channels_img, 4, 2, 1, batch_norm=False), # 64x64
            nn.Tanh()
        )

        self.embed = nn.Embedding(n_classes, embed_dim)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm=True):
        layers = []
        bias = False if batch_norm else True
        layers.append(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        )
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    
    def forward(self, x, labels):
        # Latent Vector x shape: N x z_dim x 1 x 1
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(2)
        x = torch.cat([x, embedding], dim=1)
        return self.gen(x)
    
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, H, W = 8, 3, 64, 64
    # Test Critic
    x = torch.randn((N, in_channels, H, W))
    disc = Critic(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1), f"Incorrect Critic shape output. Expected: {(N, 1, 1, 1)}. Actual: {disc(x).shape}"
    # Test Generator
    z_dim = 100
    z = torch.randn((N, z_dim, 1, 1))
    gen = Generator(z_dim, in_channels, 8)
    initialize_weights(gen)
    assert gen(z).shape == (N, in_channels, H, W), f"Incorrect Generator shape output. Expected: {(N, in_channels, H, W)}. Actual: {gen(x).shape}"

if __name__ == "__main__":
    test()