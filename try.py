import torch
from models import SimpleClassifier

model = SimpleClassifier()

total = 0
for params in model.parameters():
    total += params.numel()

print(total / 10 ** 6)
checkpoints = model.state_dict()
torch.save(checkpoints, './saved_models/check.pt')