import torch
import torch.nn as nn


class LocalEncoder(nn.Module):
    """
    Local encoder run by each party.

    Input:
        x: [B, 1, 14, 14]

    Output:
        z: [B, latent_dim]
    """

    def __init__(self, latent_dim=8):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),              # [B, 1, 14, 14] -> [B, 196]
            nn.Linear(14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class ServerClassifier(nn.Module):
    """
    Fusion center classifier.

    Input:
        z_cat: [B, 4 * latent_dim]

    Output:
        logits: [B, 10]
    """

    def __init__(self, latent_dim=8, num_classes=10):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4 * latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, z_cat):
        return self.net(z_cat)


class QuadrantVFLModel(nn.Module):
    """
    Full model:
      - 4 local encoders (one per quadrant / party)
      - 1 server-side classifier

    Inputs:
        q1, q2, q3, q4: each [B, 1, 14, 14]

    Output:
        logits: [B, 10]
    """

    def __init__(self, latent_dim=8, num_classes=10):
        super().__init__()

        self.encoder1 = LocalEncoder(latent_dim=latent_dim)
        self.encoder2 = LocalEncoder(latent_dim=latent_dim)
        self.encoder3 = LocalEncoder(latent_dim=latent_dim)
        self.encoder4 = LocalEncoder(latent_dim=latent_dim)

        self.server = ServerClassifier(
            latent_dim=latent_dim,
            num_classes=num_classes,
        )

    def forward(self, q1, q2, q3, q4):
        z1 = self.encoder1(q1)
        z2 = self.encoder2(q2)
        z3 = self.encoder3(q3)
        z4 = self.encoder4(q4)

        z_cat = torch.cat([z1, z2, z3, z4], dim=1)
        logits = self.server(z_cat)
        return logits


if __name__ == "__main__":
    model = QuadrantVFLModel(latent_dim=8, num_classes=10)

    q1 = torch.randn(4, 1, 14, 14)
    q2 = torch.randn(4, 1, 14, 14)
    q3 = torch.randn(4, 1, 14, 14)
    q4 = torch.randn(4, 1, 14, 14)

    logits = model(q1, q2, q3, q4)

    print("Output shape:", logits.shape)   # should be [4, 10]