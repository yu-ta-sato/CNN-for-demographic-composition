from utils import *


class RemoteSensingCNN(nn.Module):
    """
    A class of CNN model with transfer learning from ResNet50.
    """

    def __init__(self, out_dim, pretrained):
        super(RemoteSensingCNN, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.conv_band2_4 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.conv_band5_7 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        self.bn_band2_4 = nn.BatchNorm2d(64)
        self.bn_band5_7 = nn.BatchNorm2d(64)
        self.conv_all = nn.Conv2d(
            in_channels=64 + 64, out_channels=64, kernel_size=1, bias=False
        )
        self.resnet = models.resnet50(pretrained=pretrained).to(self.device)
        self.fc2 = nn.Linear(1000, out_dim)
        self.dropout = nn.Dropout(0.25)

        self.conv_band2_4.load_state_dict(self.resnet.conv1.state_dict())
        self.conv_band5_7.load_state_dict(self.resnet.conv1.state_dict())

    def forward(self, x):
        band2_4 = x[:, 2 - 1 : 4, :, :].flip(dims=[0])
        band5_7 = x[:, 5 - 1 : 7, :, :].flip(dims=[0])

        band2_4 = self.conv_band2_4(band2_4)
        band5_7 = self.conv_band5_7(band5_7)

        band2_4 = F.relu(self.bn_band2_4(band2_4))
        band5_7 = F.relu(self.bn_band5_7(band5_7))

        x = torch.cat((band2_4, band5_7), dim=1)
        x = self.conv_all(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.resnet.fc(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))

        return x
