import torch
import torch.nn as nn

# from plm_encoder import PLMEncoder
from transformers import BertTokenizer, AutoTokenizer, AutoModel, AutoModelForMaskedLM, RobertaModel
from subword_aggregation import SubwordAggregation
from rgat_stacks import RGATStack
from plm_encoder import PLMEncoder
from adapter import EmbeddingAdapter

# This class combines the PLM, Aggregation, and RGATStack into a single module.
class GraphEncoder(nn.Module):
    """
    This class now fully encapsulates the entire graph encoding pipeline.
    It takes the raw graph data, gets PLM embeddings, aggregates them,
    and then passes them through the RGATStack.
    """
    def __init__(self, vocab_size, embedding_dim, text_encoder_name, hidden_channels, num_layers, num_heads, num_relations, max_rows=1500, max_cols=1500):
        super(GraphEncoder, self).__init__()

        # 1. The small, learnable adapter network.
        self.adapter = EmbeddingAdapter(
            input_dim=embedding_dim,
            bottleneck_dim=hidden_channels // 2, # A common choice
            output_dim=embedding_dim
        )
        
        # 2. Create the learnable embedding layers here ---
        self.row_embedding = nn.Embedding(max_rows + 1, embedding_dim)
        self.col_embedding = nn.Embedding(max_cols + 1, embedding_dim)

        # 3. Modality embedding
        self.modality_embedding = nn.Embedding(5, embedding_dim)

        # 4. Initialize the RGATStack.
        #    The `in_channels` for the RGAT must match the output size of the PLM.
        self.rgat_stack = RGATStack(
            in_channels=embedding_dim,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            num_relations=num_relations
        )
        self.output_channels = hidden_channels

    def forward(self, graph_data):

        
        # 1. Get the pre-computed static embeddings from the data object.
        static_base_embeddings = graph_data.x
        
        # 2. Pass them through the small, learnable adapter.
        adapted_embeddings = self.adapter(static_base_embeddings)
        
        # 3. Create and add the learnable location embeddings.
        r_embed = self.row_embedding(graph_data.row_indices)
        c_embed = self.col_embedding(graph_data.col_indices)

        # 4. Create the learnable modality embeddings.
        modality_embed = self.modality_embedding(graph_data.modality_list)

        # 4. Create the final, learnable node features.
        final_node_features = adapted_embeddings + r_embed + c_embed + modality_embed
        
        # 5. Pass to the RGAT Stack.
        return self.rgat_stack(final_node_features, graph_data.edge_index, graph_data.edge_type)