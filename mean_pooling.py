import torch.nn as nn 
import torch

# study transformer representation

class MeanPooling(nn.Module):
    
    def __init__(self):
        super(MeanPooling,self).__init__()

    def forward(self,last_hidden_state: torch.Tensor, attention_mask:torch.Tensor):
        input_mask_explained = attention_mask.unsqueeze(-1).expand(last_hidden_state.shape)
        sum_embeddings = torch.sum(input_mask_explained * last_hidden_state, 1)
        sum_mask = input_mask_explained.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings




last_hidden_state = torch.tensor([
    
    [[1.0, 2.0, 3.0],   # sentence 1, token 1
     [4.0, 5.0, 6.0],   # sentence 1, token 2
     [7.0, 8.0, 9.0],   # sentence 1, token 3
     [10., 11., 12.]],  # sentence 1, token 4 (this is padding)

    [[2.0, 3.0, 4.0],   # sentence 2, token 1
     [5.0, 6.0, 7.0],   # sentence 2, token 2
     [8.0, 9.0, 10.],   # sentence 2, token 3 (this is padding)
     [11., 12., 13.]]

])

attention_mask = torch.tensor([
    [1,1,1,0], # sentence 1 3 tokens
    [1,1,0,0]  # sentence 2 2 tokens
])

# last_hidden_state -> [2,4,3]
# attention mask -> [2,4]

pooling = MeanPooling()
pooling_values = pooling(last_hidden_state, attention_mask)

print(f'Pooling values :{pooling_values}')
print(f'Pooling Shape: {pooling_values.shape}')

