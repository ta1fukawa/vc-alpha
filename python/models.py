import numpy as np
import torch

class EmbedModel1d(torch.nn.Module):
    def __init__(self, n_freq, n_frames):
        super(EmbedModel1d, self).__init__()

        self.conv1a = torch.nn.Conv1d(n_freq, 512, kernel_size=3, dilation=1, padding='same')
        self.conv1b = torch.nn.Conv1d(512, 512, kernel_size=3, dilation=1, padding='same')
        self.drop1  = torch.nn.Dropout(p=0.2)

        self.conv2a = torch.nn.Conv1d(512, 512, kernel_size=3, dilation=1, padding='same')
        self.conv2b = torch.nn.Conv1d(512, 512, kernel_size=3, dilation=1, padding='same')
        self.drop2  = torch.nn.Dropout(p=0.2)

        self.conv5  = torch.nn.Conv1d(512, 2048, kernel_size=3, dilation=1, padding='same')
        self.line5  = torch.nn.Linear(4096, 512)
            
    def _stats_pooling(self, x):
        mean = torch.mean(x, dim=2)
        std = torch.sqrt(torch.clamp(torch.var(x, dim=2), 1e-8, None))
        return torch.cat([mean, std], dim=1)

    def forward(self, x):
        x = torch.permute(x, dims=[0, 2, 1])

        x = torch.nn.functional.relu(self.conv1a(x))
        x = torch.nn.functional.relu(self.conv1b(x))
        x = self.drop1(x)
        
        x = torch.nn.functional.relu(self.conv2a(x))
        x = torch.nn.functional.relu(self.conv2b(x))
        x = self.drop2(x)
        
        x = self.conv5(x)
        x = self._stats_pooling(x)
        x = self.line5(x)

        return x

class EmbedModel2d(torch.nn.Module):
    def __init__(self, n_freq, n_frames):
        super(EmbedModel2d, self).__init__()
        
        self.conv1a = torch.nn.Conv2d(1, 64, kernel_size=(5, 5), dilation=(1, 1), padding='same')
        self.conv1b = torch.nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 1), padding='same')
        self.drop1  = torch.nn.Dropout2d(p=0.2)
        self.pool1a = torch.nn.MaxPool2d(kernel_size=(1, 4))
        self.pool1b = torch.nn.AvgPool2d(kernel_size=(1, 4))
        
        self.conv2a = torch.nn.Conv2d(128, 128, kernel_size=(5, 5), dilation=(1, 1), padding='same')
        self.conv2b = torch.nn.Conv2d(128, 128, kernel_size=(5, 5), dilation=(1, 1), padding='same')
        self.drop2  = torch.nn.Dropout2d(p=0.2)
        self.pool2a = torch.nn.MaxPool2d(kernel_size=(1, 4))
        self.pool2b = torch.nn.AvgPool2d(kernel_size=(1, 4))
        
        self.conv3a = torch.nn.Conv2d(256, 256, kernel_size=(5, 5), dilation=(1, 1), padding='same')
        self.conv3b = torch.nn.Conv2d(256, 256, kernel_size=(5, 5), dilation=(1, 1), padding='same')
        self.drop3  = torch.nn.Dropout2d(p=0.2)
        self.pool3a = torch.nn.MaxPool2d(kernel_size=(1, 4))
        self.pool3b = torch.nn.AvgPool2d(kernel_size=(1, 4))
        
        self.conv7  = torch.nn.Conv2d(512, 2048, kernel_size=(5, 5), dilation=(1, 1), padding='same')
        self.line7  = torch.nn.Linear(4096, 512)

    def _stats_pooling(self, x):
        mean = torch.mean(x, dim=[2, 3])
        std = torch.sqrt(torch.clamp(torch.var(x, dim=[2, 3]), 1e-8, None))
        return torch.cat([mean, std], dim=1)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)

        x = torch.nn.functional.relu(self.conv1a(x))
        x = torch.nn.functional.relu(self.conv1b(x))
        x = self.drop1(x)
        x = torch.cat([self.pool1a(x), self.pool1b(x)], dim=1)
        
        x = torch.nn.functional.relu(self.conv2a(x))
        x = torch.nn.functional.relu(self.conv2b(x))
        x = self.drop2(x)
        x = torch.cat([self.pool2a(x), self.pool2b(x)], dim=1)
        
        x = torch.nn.functional.relu(self.conv3a(x))
        x = torch.nn.functional.relu(self.conv3b(x))
        x = self.drop1(x)
        x = torch.cat([self.pool3a(x), self.pool3b(x)], dim=1)
        
        x = self.conv7(x)
        x = torch.reshape(x, [x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]])
        x = self.line7(x)

        return x

class FullModel(torch.nn.Module):
    def __init__(self, dim, n_freq=512, n_frames=32, nclasses=16):
        super(FullModel, self).__init__()
        
        if dim == 1:
            self.embed = EmbedModel1d(n_freq, n_frames)
        elif dim == 2:
            self.embed = EmbedModel2d(n_freq, n_frames)
        else:
            raise ValueError('引数dimは1～2である必要があります。')

        self.drop1 = torch.nn.Dropout(p=0.2)

        self.line2 = torch.nn.Linear(512, 512)
        self.drop2 = torch.nn.Dropout(p=0.2)

        self.line3 = torch.nn.Linear(512, nclasses)

    def forward(self, x):
        x = torch.nn.functional.relu(self.embed(x))
        x = self.drop1(x)

        x = torch.nn.functional.relu(self.line2(x))
        x = self.drop2(x)

        x = torch.nn.functional.log_softmax(self.line3(x), dim=-1)

        return x
