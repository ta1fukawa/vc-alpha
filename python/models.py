import numpy as np
import torch

class EmbedModel1d(torch.nn.Module):
    def __init__(self, sp_length):
        super(EmbedModel1d, self).__init__()
        self.conv1a = torch.nn.Conv1d(sp_length, 512, kernel_size=3, stride=1, dilation=1, padding='same')
        self.conv1b = torch.nn.Conv1d(512, 512, kernel_size=3, stride=1, dilation=1, padding='same')
        self.drop1  = torch.nn.Dropout(p=0.2)
        self.conv2a = torch.nn.Conv1d(512, 512, kernel_size=3, stride=1, dilation=2, padding='same')
        self.conv2b = torch.nn.Conv1d(512, 512, kernel_size=3, stride=1, dilation=1, padding='same')
        self.drop2  = torch.nn.Dropout(p=0.2)
        self.conv3a = torch.nn.Conv1d(512, 512, kernel_size=3, stride=1, dilation=3, padding='same')
        self.conv3b = torch.nn.Conv1d(512, 512, kernel_size=3, stride=1, dilation=1, padding='same')
        self.drop3  = torch.nn.Dropout(p=0.2)
        self.conv4a = torch.nn.Conv1d(512, 512, kernel_size=3, stride=1, dilation=1, padding='same')
        self.conv4b = torch.nn.Conv1d(512, 512, kernel_size=3, stride=1, dilation=1, padding='same')
        self.drop4  = torch.nn.Dropout(p=0.2)
        self.conv5  = torch.nn.Conv1d(512, 2048, kernel_size=3, stride=1, dilation=1, padding='same')
        self.drop5  = torch.nn.Dropout(p=0.2)
        self.line1  = torch.nn.Linear(4096, 512)

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
        x = torch.nn.functional.relu(self.conv3a(x))
        x = torch.nn.functional.relu(self.conv3b(x))
        x = self.drop3(x)
        x = torch.nn.functional.relu(self.conv4a(x))
        x = torch.nn.functional.relu(self.conv4b(x))
        x = self.drop4(x)
        x = self._stats_pooling(self.conv5(x))
        x = self.drop5(x)
        x = self.line1(x)

        return x

class EmbedModel2d(torch.nn.Module):
    def __init__(self):
        super(EmbedModel2d, self).__init__()
        self.conv1a = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.conv1b = torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.drop1  = torch.nn.Dropout2d(p=0.2)
        self.pool1  = torch.nn.MaxPool2d(kernel_size=(2, 4))
        
        self.conv2a = torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.conv2b = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.drop2  = torch.nn.Dropout2d(p=0.2)
        self.pool2  = torch.nn.MaxPool2d(kernel_size=(2, 4))
        
        self.conv3a = torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.conv3b = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.drop3  = torch.nn.Dropout2d(p=0.2)
        self.pool3  = torch.nn.MaxPool2d(kernel_size=(2, 4))
        
        self.conv4  = torch.nn.Conv2d(128, 2048, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.line4  = torch.nn.Linear(4096, 512)

    def _stats_pooling(self, x):
        mean = torch.mean(x, dim=[2, 3])
        std = torch.sqrt(torch.clamp(torch.var(x, dim=[2, 3]), 1e-8, None))
        return torch.cat([mean, std], dim=1)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)

        x = torch.nn.functional.relu(self.conv1a(x))
        x = torch.nn.functional.relu(self.conv1b(x))
        x = self.pool1(self.drop1(x))
        
        x = torch.nn.functional.relu(self.conv2a(x))
        x = torch.nn.functional.relu(self.conv2b(x))
        x = self.pool2(self.drop2(x))
        
        x = torch.nn.functional.relu(self.conv3a(x))
        x = torch.nn.functional.relu(self.conv3b(x))
        x = self.pool3(self.drop3(x))

        x = self._stats_pooling(self.conv4(x))
        x = self.line4(x)

        return x

class FullModel(torch.nn.Module):
    def __init__(self, dim, sp_length, nclasses):
        super(FullModel, self).__init__()
        
        if dim == 1:
            self.embed = EmbedModel1d(sp_length)
        elif dim == 2:
            self.embed = EmbedModel2d()
        else:
            raise ValueError('引数dimは1～2である必要があります。')

        self.drop6  = torch.nn.Dropout(p=0.2)
        self.line2 = torch.nn.Linear(512, 512)
        self.drop7  = torch.nn.Dropout(p=0.2)
        self.line3 = torch.nn.Linear(512, nclasses)

    def forward(self, x):
        x = torch.nn.functional.relu(self.embed(x))
        x = self.drop6(x)
        x = torch.nn.functional.relu(self.line2(x))
        x = self.drop7(x)
        x = torch.nn.functional.log_softmax(self.line3(x), dim=-1)

        return x