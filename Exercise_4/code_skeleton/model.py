import torch.nn as nn



class ResBlock(nn.Module):
    def __init__(self, insize, outsize, stride):
        super(ResBlock, self).__init__()
        self.layers_resblock = nn.Sequential(
            nn.Conv2d(in_channels=insize, out_channels=outsize, kernel_size=3, stride=stride),
            nn.BatchNorm2d(outsize),
            nn.ReLU(),
            nn.Conv2d(in_channels=outsize, out_channels=outsize, kernel_size=3),
            nn.BatchNorm2d(outsize)
        )

    def forward(self, x):
        out = self.layers_resblock(x)
        return out


class ResNet(nn.Module):
    def __init__(self, outsize=64):
        super(ResNet, self).__init__()

        self.resblock1 = ResBlock(insize=64, outsize=64, stride=1)
        self.resblock2 = ResBlock(insize=64, outsize=128, stride=2)
        self.resblock3 = ResBlock(insize=128, outsize=256, stride=2)
        self.resblock4 = ResBlock(insize=256, outsize=512, stride=2)

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=outsize, kernel_size=(7, 7), stride=(2, 2)),
            nn.BatchNorm2d(outsize),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.resblock1,
            self.resblock2,
            self.resblock3,
            self.resblock4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        return x
