"""Module for the TacNet"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvolution(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_convolution = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, network_input):
        return self.double_convolution(network_input)


class Encoder(nn.Module):
    """Double convolution then downscaling"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_convolution = nn.Sequential(
            nn.MaxPool2d(2), DoubleConvolution(in_channels, out_channels)
        )

    def forward(self, network_input):
        return self.maxpool_convolution(network_input)


class Decoder(nn.Module):
    """UP (upscaling) convolution then double convolution"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.upsampling_layer = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            self.convolutional_layer = DoubleConvolution(
                in_channels, out_channels, in_channels // 2
            )
        else:
            self.upsampling_layer = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.convolutional_layer = DoubleConvolution(in_channels, out_channels)

    def forward(self, signal_1, signal_2):
        signal_1 = self.upsampling_layer(signal_1)
        # input shape is B, C, H, W
        height_difference = signal_2.size()[2] - signal_1.size()[2]
        width_difference = signal_2.size()[3] - signal_1.size()[3]

        signal_2 = F.max_pool2d(
            signal_2,
            kernel_size=(height_difference + 1, width_difference + 1),
            stride=1,
            padding=0,
        )

        concatenated_signals = torch.cat([signal_2, signal_1], dim=1)
        return self.convolutional_layer(concatenated_signals)


class OutConv(nn.Module):
    """Map features to the three-component (3-channel) forces"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, input_features):
        return self.convolution(input_features)


import torch
import torch.nn as nn
import torch.nn.functional as F

class TacNet(nn.Module):
    """TacNet Model Definition"""
    def __init__(self, in_nc=1, num_of_neurons=2048, bilinear=False):
        super(TacNet, self).__init__()
        self.inc = DoubleConvolution(in_nc, 8)
        self.down1 = Encoder(8, 16)
        self.down2 = Encoder(16, 32)
        self.down3 = Encoder(32, 64)
        self.down4 = Encoder(64, 128)
        self.down5 = Encoder(128, 256)
        self.down6 = nn.Conv2d(256, 256, kernel_size=(3, 2), padding=0)
        factor = 2 if bilinear else 1
        self.up1 = Decoder(256, 128 // factor, bilinear)
        self.up2 = Decoder(128, 64 // factor, bilinear)
        self.outc = OutConv(64, 3)

        # Compute the flattened size with the updated input size (256x256)
        self.flattened_size = self._compute_flattened_size()
        # print(f"Flattened size: {self.flattened_size}")  # Debugging: Print the flattened size

        # Corrected the input size of self.fc1
        self.fc1 = nn.Linear(10752, num_of_neurons)
        self.fc2 = nn.Linear(num_of_neurons, num_of_neurons)
        self.fc3 = nn.Linear(num_of_neurons, 4)  # 4 outputs for regression
        self.lrelu = nn.LeakyReLU(0.02, inplace=True)

    def _compute_flattened_size(self):
        """Compute the size of the flattened output after the convolutional layers."""
        # Pass a dummy input through the convolutional layers to get the output size
        dummy_input = torch.zeros(1, 1, 256, 256)  # Updated to (1, 1, 256, 256)
        x = self.inc(dummy_input)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.down6(x)
        # print(f"Shape after down6: {x.shape}")  # Debugging: Print shape after down6
        return x.view(-1).size(0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        # print(f"Shape before flattening: {x7.shape}")  # Debugging: Print shape before flattening

        x = x7.view(x7.size(0), -1)
        # print(f"Shape after flattening: {x.shape}")  # Debugging: Print shape after flattening

        # Pass through fully connected layers
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x = self.fc3(x)
        return x


class TacNetMassageArm(nn.Module):
    """Class for the TacNet Model"""

    def __init__(self, in_nc = 2, 
                       num_of_features = 585, 
                       num_of_neurons = 2048,
                       n_classes = 4,  
                       bilinear=False):
        super().__init__()
        self.n_channels = in_nc
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.num_of_neurons = num_of_neurons

        self.inc = DoubleConvolution(in_nc, 8)
        self.down1 = Encoder(8, 16)
        self.down2 = Encoder(16, 32)
        self.down3 = Encoder(32, 64)
        self.down4 = Encoder(64, 128)
        self.down5 = Encoder(128, 256)
        # formulation of 2d convolutinon with stride=1, padding=0
        # Hout = Hin - kernel_size + 1 (= Hin - 2)
        # Wout = Win - kernel_size + 1 (= Win - 1)
        self.down6 = nn.Conv2d(256, 256, kernel_size=(3, 2), padding=0)
        factor = 2 if bilinear else 1
        self.up1 = Decoder(256, 128 // factor, bilinear)
        self.up2 = Decoder(128, 64 // factor, bilinear)
        self.outc = OutConv(64, n_classes)

        outputs = num_of_features * 4
        # self.fc1 = nn.Linear(3 * 24 * 28, self.num_of_neurons, bias=True)
        self.fc1 = nn.Linear(3 * 52 * 56, self.num_of_neurons, bias=True)
        self.fc2 = nn.Linear(self.num_of_neurons, self.num_of_neurons, bias=True)
        self.fc3 = nn.Linear(self.num_of_neurons, outputs, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, input_signal):  # x:   6, 256, 256
        
        encoded = self.inc(input_signal)  # x1:  8, 256, 256
        # print(x1.shape)
        encoded = self.down1(encoded)  # x2: (16, 128, 128)
        # print(x2.shape)
        encoded = self.down2(encoded)  # x3: (32, 64, 64)
        # print(x3.shape)
        encoded = self.down3(encoded)  # x4: (64, 32, 32)
        fourth_encoded = encoded
        # print(x4.shape)
        encoded = self.down4(encoded)  # x5: (128, 16, 16)
        fifth_encoded = encoded
        # print(x5.shape)
        encoded = self.down5(encoded)  # x6: (256, 8, 8)
        # print(x6.shape)
        seventh_encoded = self.down6(encoded)  # x7: (256, 6, 7)
        # print(x7.shape)
        decoded = self.up1(seventh_encoded, fifth_encoded)  # x:  (128, 12, 14)
        # print(x.shape)
        decoded = self.up2(decoded, fourth_encoded)  # x:  (64, 24, 28)
        # print(x.shape)
        force_map = self.outc(decoded)  # force_map:  (3, 24, 28)
        # print(force_map.shape)
        # feed forward fully connected layers for x feature map (channel)
        num_of_fore_map_features = self.number_flat_features(force_map)
        output = force_map.view(-1, num_of_fore_map_features)
        output = self.lrelu(self.fc1(output))
        output = self.lrelu(self.fc2(output))
        output = self.fc3(output)


        return output

    def number_flat_features(self, features):
        feature_size = features.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for size in feature_size:
            num_features *= size
        return num_features
    
class TacNetMarker(nn.Module):
    """Class for the TacNet Model"""
    def __init__(self, 
                 in_nc=2, 
                 num_of_features = 585, 
                 num_of_neurons= 1024, 
                 n_classes=3, 
                 bilinear=False):
        super().__init__()
        self.n_channels = in_nc
        self.n_classes = n_classes
        self.num_of_neurons = num_of_neurons

        outputs = num_of_features
        self.fc1 = nn.Linear(self.n_channels * 5 * 6, self.num_of_neurons, bias=True)
        self.fc2 = nn.Linear(self.num_of_neurons, self.num_of_neurons, bias=True)
        self.fc3 = nn.Linear(self.num_of_neurons, outputs, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, input_signal):  # x: (2, 5, 6)
        num_of_input_features = self.number_flat_features(input_signal)
        output = input_signal.reshape(-1, num_of_input_features)
        
        output = self.lrelu(self.fc1(output))
        output = self.lrelu(self.fc2(output))
        output = self.fc3(output)

        return output

    def number_flat_features(self, features):
        feature_size = features.size()[1:]
        num_features = 1
        for size in feature_size:
            num_features *= size
        return num_features