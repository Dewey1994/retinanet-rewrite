import  torch
import torch.nn as nn
# a = torch.randn([1,3,64,64])
# x = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
# y = torch.nn.Sequential(nn.Conv2d(3,64, kernel_size=3, stride=1, padding=1),
#                         nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1),
#                         nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1))
# b = x(a)
# print(b.shape)
# c = y(a)
# print(c.shape)

a = nn.Conv2d(256, 256, 3)
a = nn.Conv2d(512, 512, 3)
print(a)
