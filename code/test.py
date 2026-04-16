import torch

mask = torch.ones([1, 2, 2, 3], dtype=torch.bool)

sub_mask = torch.tensor([[True, False]])

sub_mask_expanded = sub_mask.unsqueeze(-1).unsqueeze(-1)

print(mask)
print(mask & sub_mask_expanded)