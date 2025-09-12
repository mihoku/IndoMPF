import torch
import torch.nn as nn
import itertools
import torch.nn.functional as F

# PyTorch Geometric is a popular library for deep learning on graphs.
# Its core message passing class is used to build custom RGAT layer.
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

# --- PyTorch Geometric RGAT Model Implementation ---

class FFN(nn.Module):
    """
    A standard Feed-Forward Network block, as described in "Attention Is All You Need".
    This is the second main sub-layer within a Transformer block. Its role is to
    process the output of the attention layer in a non-linear way.
    """
    def __init__(self, model_dim, ffn_dim, dropout=0.2):
        super(FFN, self).__init__()
        # The "expand-and-contract" pattern.
        # The first linear layer expands the feature dimension.
        self.linear1 = nn.Linear(model_dim, ffn_dim)
        # The second linear layer contracts it back to the original dimension.
        self.linear2 = nn.Linear(ffn_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        # GELU is a modern activation function commonly used in Transformers (like BERT).
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class RGATBlock(MessagePassing):
    """
    A self-contained Relational Graph Attention Transformer Block.
    This class encapsulates one full layer of the graph transformer, which includes
    both the attention mechanism and the feed-forward network.
    Inherits from PyTorch Geometric's MessagePassing class to handle graph structure.
    """
    def __init__(self, channels, num_relations, num_heads=8, ffn_dim_multiplier=4, normalization_layer = 'rms_norm', **kwargs):
        
        # Specify 'add' aggregation during initialization, meaning messages from neighbors will be summed up.
        # aggr argument is inherited directly from the MessagePassing base class.
        # It tells the MessagePassing framework how to combine all the incoming messages for a single node into one vector. 
        # Since a node can have many neighbors sending messages simultaneously, we need a function to aggregate them.
        super(RGATBlock, self).__init__(aggr='add', node_dim=0, **kwargs)
        
        # channels represent feature dimension of each node
        # It's the length of the vector that represents a single node in the graph.
        # channels is the standard terminology used throughout the PyTorch Geometric library 
        # and is borrowed from Convolutional Neural Networks (CNNs), where it refers to the depth of a feature map. 
        # PyG uses it to maintain a consistent API.
        self.channels = channels
        
        # the number of attention heads in the multi-head attention mechanism.
        # Each head will learn its own set of attention weights.
        self.num_heads = num_heads

        # Dimension of each individual attention head
        # In multi-head attention, the model doesn't just calculate attention once. 
        # It performs the attention mechanism multiple times in parallel, and each parallel instance is called a "head".
        # Therefore, we take the main feature vector for a node (which has a length of channels) 
        # and split it into smaller, equal-sized chunks. 
        # Each chunk is then processed by a separate "attention head."
        # This is the dimension of the Query, Key, and Value vectors for each head.
        # It's calculated as channels divided by the number of heads.
        self.heads_dim = channels // num_heads 

        # --- Attention Sub-layer Components ---
        # These are the W_q, W_k, and W_v matrices from the paper, used to project
        # the input node features into Query, Key, and Value spaces.
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        
        # This embedding layer learns a unique vector for each type of relation.
        # num_relations is the number of unique edge types.
        self.relation_embed = nn.Embedding(num_relations, channels)
        
        # This final linear layer combines the outputs from all attention heads.
        self.out_proj = nn.Linear(channels, channels)

        # Normalization for the attention sub-layer.
        try:
            # If the normalization_layer is 'layer_norm', we use LayerNorm.
            # If it is 'rms_norm', we use RMSNorm.
            # This allows for flexibility in choosing the normalization technique.
            assert normalization_layer in ['layer_norm', 'rms_norm']
        except AssertionError:
            raise ValueError("normalization_layer must be either 'layer_norm' or 'rms_norm'.")
        # Depending on the normalization_layer argument, we initialize the appropriate normalization layer.
        if normalization_layer == 'layer_norm':
            # LayerNorm is a normalization technique that normalizes the input across the feature dimension.
            # It stabilizes the learning process by ensuring that the inputs to each layer have a consistent
            self.norm1 = nn.LayerNorm(channels)
        elif normalization_layer == 'rms_norm':
            # RMSNorm is a variant of layer normalization that normalizes the input by its root mean square.
            # It is often used in Transformer models as an alternative to LayerNorm.
            # It is less computationally expensive than LayerNorm.
            self.norm1 = nn.RMSNorm(channels)
        
        self.dropout1 = nn.Dropout(0.2)

        # --- FFN Sub-layer Components ---
        # ffn_dim_multiplier is a hyperparameter that controls the size of the hidden layer within the Feed-Forward Network (FFN). 
        # It determines how much the FFN "expands" the feature dimension.
        # The value of 4 is the standard default from the original "Attention Is All You Need" paper, although it's not always optimal.
        self.ffn = FFN(channels, channels * ffn_dim_multiplier)

        # Layer normalization for the FFN sub-layer.
        # This is applied after the FFN to stabilize the outputs.
        # Depending on the normalization_layer argument, we initialize the appropriate normalization layer.
        if normalization_layer == 'layer_norm':
            # LayerNorm is a normalization technique that normalizes the input across the feature dimension.
            # It stabilizes the learning process by ensuring that the inputs to each layer have a consistent
            self.norm2 = nn.LayerNorm(channels)
        elif normalization_layer == 'rms_norm':
            # RMSNorm is a variant of layer normalization that normalizes the input by its root mean square.
            # It is often used in Transformer models as an alternative to LayerNorm.
            # It is less computationally expensive than LayerNorm.
            self.norm2 = nn.RMSNorm(channels)

        # Dropout for the FFN output to prevent overfitting.
        # This is applied after the FFN output.
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_type):
        # --- 1. Relational Graph Attention Sub-layer ---
        # This follows the "Pre-LN" architecture where normalization is applied before the operation.
        residual = x # Store the input for the residual connection.
        x_norm = self.norm1(x)

        # Project inputs into Q, K, V ando get relation embeddings.
        # The `.view()` reshapes the tensr for multi-head attention.
        # It returns a new tensor that is a "view" of the original tensor, meaning they share the same memory.
        # The `view` operation reshapes the tensor to have a shape of (num_nodes, num_heads, heads_dim).
        # When we do q = self.query(x_norm), we are taking the feature vector x_norm for a node and passing it 
        # through a learnable transformation to create its "Query" personality for the attention calculation. 
        # The model learns the best weights for the self.query layer to produce queries 
        # that are effective at finding relevant information from other nodes. 
        # The same happens for the Key and Value layers.
        q = self.query(x_norm).view(-1, self.num_heads, self.heads_dim)
        k = self.key(x_norm).view(-1, self.num_heads, self.heads_dim)
        v = self.value(x_norm).view(-1, self.num_heads, self.heads_dim)
        r = self.relation_embed(edge_type).view(-1, self.num_heads, self.heads_dim)
        
        # `propagate` is the main function from PyG's MessagePassing class.
        # It handles the message passing loop over the graph edges efficiently.
        # This is the main call to the MessagePassing framework.
        # It will internally call the `message()` function for each edge.
        attn_out = self.propagate(edge_index, q=q, k=k, v=v, r=r)
        # Reshape the multi-head output back into a single vector per node
        # before passing it to the final projection layer.
        # Shape: [num_nodes, num_heads, heads_dim] -> [num_nodes, channels]
        attn_out = attn_out.view(-1, self.num_heads * self.heads_dim)
        attn_out = self.out_proj(attn_out)

        # Apply the first residual connection ("Add & Norm" from the paper).
        x = residual + self.dropout1(attn_out)

        # --- 2. Feed-Forward Network Sub-layer ---
        residual = x # Store the output of the attention layer for the next residual connection.
        x_norm = self.norm2(x) # Apply Pre-LayerNorm again.
        ffn_out = self.ffn(x_norm)

        # Apply the second residual connection.
        x = residual + self.dropout2(ffn_out)
        
        # RGAT block returns the new, updated node feature tensor x.
        # The fundamental job of a graph convolutional layer like RGATBlock 
        # is to update the features of the nodes, not to change the structure of the graph itself.
        return x

    def message(self, q_i, k_j, v_j, r, index, ptr, size_i):
        # This function defines the "message" sent from a source node (j) to a target node (i).
        # It's where the core attention logic happens.
        
        # This is the "Relational" part: modify the Key and Value with the relation embedding.
        k_j = k_j + r
        v_j = v_j + r
        
        # Calculate scaled dot-product attention score, as per the paper.
        attention_score = (q_i * k_j).sum(dim=-1) / (self.heads_dim ** 0.5)
        
        # Normalize scores using softmax. `softmax` from PyG is aware of the graph structure
        # and correctly normalizes scores for each node's incoming edges.
        # It computes the attention probabilities for each edge.
        # `index` is the target node indices, `ptr` is used for batching,
        # and `size_i` is the number of nodes in the target set.
        # This ensures that the attention scores are normalized across all neighbors of each node.
        attention_probs = softmax(attention_score, index, ptr, size_i)
        
        # The final message is the value vector weighted by its attention probability.
        # This is the actual message that will be sent to the target node.
        # The `unsqueeze(-1)` adds a new dimension to the attention probabilities because it is currently 2 dimensinonal,
        # allowing it to be multiplied with the value vectors which is 3 dimensional.
        return v_j * attention_probs.unsqueeze(-1)


class RGATStack(nn.Module):
    """
    The top-level model, which is a stack of our self-contained RGATBlock layers.
    """
    #def __init__(self, hidden_channels, num_layers, num_heads, num_relations):
    def __init__(self, in_channels, hidden_channels, num_layers, num_heads, num_relations):   
        super(RGATStack, self).__init__()
        
        # An initial linear layer to project the input features (e.g., from BERT)
        # to the model's hidden dimension.
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # Create a stack of RGATBlock layers.
        # Each block will process the graph data and learn to extract features.
        # num_layers is the number of RGATBlock layers in the model.
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(RGATBlock(hidden_channels, num_relations, num_heads))

    def forward(self, x, edge_index, edge_type):

        # Apply the initial projection.
        x = self.input_proj(x)
        
        # Sequentially pass the node features through each Transformer block.
        # The output of one block becomes the input to the next.
        for block in self.blocks:

            # x is a tensor that holds the current feature vector for every node in the graph. 
            # Initially, it contains the basic embeddings, but it gets updated in every step.            
            # The RGATBlock takes in the graph structure as a guide to know how to update x. 
            # It uses the edge_index and edge_type as read-only instructions, but it doesn't modify them.
            # edge_index is a tensor that defines the connections between nodes in the graph.
            # edge_type is a tensor that specifies the type of relation for each edge.
            # The edge_index and edge_type are passed into every block, but they are never reassigned. 
            # Only x is updated in each iteration.
            # It is directly calling the forward method of the RGATBlock instance that the loop is currently on.
            x = block(x, edge_index, edge_type)

        return x


