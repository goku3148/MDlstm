import torch
import torch.nn as nn

x = torch.randint(5,20,(4,2))
y = torch.randint(1,5,(4,2))

print(x)
print(y)
print(x + y)