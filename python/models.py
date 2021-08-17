import numpy as np
import torch

class EmbedModel1d(torch.nn.Module):
    def __init__(self, sp_length, kernel_size):
        super(EmbedModel1d, self).__init__()

        self.conv1a = torch.nn.Conv1d(sp_length, 512, kernel_size=3, stride=1, dilation=1, padding='same')
        self.conv1b = torch.nn.Conv1d(512, 512, kernel_size=3, stride=1, dilation=1, padding='same')
        self.drop1  = torch.nn.Dropout(p=0.2)
        self.pool1  = torch.nn.MaxPool1d(kernel_size=kernel_size)

        self.conv2a = torch.nn.Conv1d(512, 512, kernel_size=3, stride=1, dilation=2, padding='same')
        self.conv2b = torch.nn.Conv1d(512, 512, kernel_size=3, stride=1, dilation=1, padding='same')
        self.drop2  = torch.nn.Dropout(p=0.2)
        self.pool2  = torch.nn.MaxPool1d(kernel_size=kernel_size)

        self.conv3a = torch.nn.Conv1d(512, 1024, kernel_size=3, stride=1, dilation=3, padding='same')
        self.conv3b = torch.nn.Conv1d(1024, 1024, kernel_size=3, stride=1, dilation=1, padding='same')
        self.drop3  = torch.nn.Dropout(p=0.2)
        self.pool3  = torch.nn.MaxPool1d(kernel_size=kernel_size)

        self.conv4a = torch.nn.Conv1d(1024, 1024, kernel_size=3, stride=1, dilation=1, padding='same')
        self.conv4b = torch.nn.Conv1d(1024, 1024, kernel_size=3, stride=1, dilation=1, padding='same')
        self.drop4  = torch.nn.Dropout(p=0.2)
        self.pool4  = torch.nn.MaxPool1d(kernel_size=kernel_size)

        self.conv5  = torch.nn.Conv1d(1024, 2048, kernel_size=3, stride=1, dilation=1, padding='same')
        self.drop5  = torch.nn.Dropout(p=0.2)
        self.line5  = torch.nn.Linear(4096, 512)

    def _stats_pooling(self, x):
        mean = torch.mean(x, dim=2)
        std = torch.sqrt(torch.clamp(torch.var(x, dim=2), 1e-8, None))
        return torch.cat([mean, std], dim=1)

    def forward(self, x):
        x = torch.permute(x, dims=[0, 2, 1])

        x = torch.nn.functional.relu(self.conv1a(x))
        x = torch.nn.functional.relu(self.conv1b(x))
        x = self.pool1(self.drop1(x))
        
        x = torch.nn.functional.relu(self.conv2a(x))
        x = torch.nn.functional.relu(self.conv2b(x))
        x = self.pool2(self.drop2(x))
        
        x = torch.nn.functional.relu(self.conv3a(x))
        x = torch.nn.functional.relu(self.conv3b(x))
        x = self.pool3(self.drop3(x))
        
        x = torch.nn.functional.relu(self.conv4a(x))
        x = torch.nn.functional.relu(self.conv4b(x))
        x = self.pool4(self.drop4(x))

        x = self._stats_pooling(self.conv5(x))
        x = self.line5(self.drop5(x))

        return x

class EmbedModel2d(torch.nn.Module):
    def __init__(self, kernel_size=(2, 4)):
        super(EmbedModel2d, self).__init__()
        
        self.conv1a = torch.nn.Conv2d(1, 128, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding='same')
        self.conv1b = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding='same')
        self.drop1  = torch.nn.Dropout2d(p=0.2)
        self.pool1  = torch.nn.MaxPool2d(kernel_size=kernel_size)
        
        self.conv2a = torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), dilation=(2, 1), padding='same')
        self.conv2b = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 2), padding='same')
        self.drop2  = torch.nn.Dropout2d(p=0.2)
        self.pool2  = torch.nn.MaxPool2d(kernel_size=kernel_size)
        
        self.conv3a = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), dilation=(3, 1), padding='same')
        self.conv3b = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 3), padding='same')
        self.drop3  = torch.nn.Dropout2d(p=0.2)
        self.pool3  = torch.nn.MaxPool2d(kernel_size=kernel_size)
        
        self.conv4a = torch.nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding='same')
        self.conv4b = torch.nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding='same')
        self.drop4  = torch.nn.Dropout2d(p=0.2)
        self.pool4  = torch.nn.MaxPool2d(kernel_size=kernel_size)
        
        self.conv5  = torch.nn.Conv2d(1024, 2048, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding='same')
        self.drop5  = torch.nn.Dropout(p=0.2)
        self.line5  = torch.nn.Linear(4096, 512)

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
        
        x = torch.nn.functional.relu(self.conv4a(x))
        x = torch.nn.functional.relu(self.conv4b(x))
        x = self.pool4(self.drop4(x))
        
        x = self._stats_pooling(self.conv5(x))
        x = self.line5(self.drop5(x))

        return x

class FullModel(torch.nn.Module):
    def __init__(self, dim, sp_length=512, nclasses=16, kernel_size=(2, 4)):
        super(FullModel, self).__init__()
        
        if dim == 1:
            self.embed = EmbedModel1d(sp_length, kernel_size[0])
        elif dim == 2:
            self.embed = EmbedModel2d(kernel_size)
        else:
            raise ValueError('引数dimは1～2である必要があります。')

        self.line1 = torch.nn.Linear(512, 512)
        self.drop1 = torch.nn.Dropout(p=0.2)

        self.line2 = torch.nn.Linear(512, nclasses)

    def forward(self, x):
        x = self.embed(x)

        x = torch.nn.functional.relu(self.line1(x))
        x = self.drop1(x)

        x = torch.nn.functional.log_softmax(self.line2(x), dim=-1)

        return x
