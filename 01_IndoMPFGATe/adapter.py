import torch.nn as nn

class EmbeddingAdapter(nn.Module):
    """
    A small, learnable network to adapt frozen embeddings.
    It takes the powerful but static embeddings from a large LLM and
    learns a task-specific transformation on them.
    """
    def __init__(self, input_dim, bottleneck_dim, output_dim):
        super(EmbeddingAdapter, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, output_dim)
        )

    def forward(self, x):
        return self.adapter(x)