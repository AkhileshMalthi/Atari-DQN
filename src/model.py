import torch.nn as nn

class AtariCNN(nn.Module):
    """
    Standard Nature DQN CNN architecture.
    METHODOLOGY:
    - Three convolutional layers extract spatial features (ball trajectory, paddle position).
    - Fully connected layers map these features to action Q-values.
    - Input shape: (4, 84, 84), representing 4 stacked grayscale frames.
    """
    def __init__(self, action_dim):
        super(AtariCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        # Scale pixels to [0, 1] for normalization/stability
        x = x.float() / 255.0
        x = self.features(x)
        x = x.reshape(x.size(0), -1) # Flatten for FC layers
        return self.fc(x)