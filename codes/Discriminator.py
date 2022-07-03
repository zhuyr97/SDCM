import torch.nn as nn

class Discriminator_SG(nn.Module):
    def __init__(self,channel=64):
        super(Discriminator_SG,self).__init__()
        channel =channel
        self.net = nn.Sequential(
            nn.Conv2d(7, channel, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(0.2),

            nn.Conv2d(channel, channel*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(channel*2, channel*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channel*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(channel*2, channel*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel*4),
            nn.LeakyReLU(0.2),


            nn.Conv2d(channel*4, channel*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel*8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(channel*8, channel*8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channel*8),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel*8, channel*16, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel*16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
