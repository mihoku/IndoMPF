import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Subword Aggregation Module ---
# This module is not part of the data reader, but it's essential for the model.
# It shows how to use the `word_to_subword_mapping` from the `Data` object.
class SubwordAggregation(nn.Module):
    def __init__(self, hidden_size, pooling_strategy='attentive'):
        super(SubwordAggregation, self).__init__()
        self.pooling_strategy = pooling_strategy

        # --- Attentive Pooling Components ---
        # This is a small, learnable attention mechanism. It's only used if the
        # pooling_strategy is set to 'attentive'.
        if self.pooling_strategy == 'attentive':
            # A linear layer to project the subword embeddings.
            self.attention_proj = nn.Linear(hidden_size, hidden_size)
            # A final layer to reduce the projected embeddings to a single score.
            self.attention_scorer = nn.Linear(hidden_size, 1)

    def forward(self, subword_embeddings, word_to_subword_mapping):
        """
        Aggregates subword embeddings into word embeddings.

        Args:
            subword_embeddings (Tensor): The output of the PLM.
                                       Shape: [num_subwords, hidden_size].
            word_to_subword_mapping (Tensor): The mapping from the Data object.
                                              Shape: [num_words, 2].
        
        Returns:
            Tensor: The final word-level node features. Shape: [num_words, hidden_size].
        """
        word_embeddings = []
        for start, end in word_to_subword_mapping:
            # For each word, select its corresponding slice of subword embeddings.
            subword_slice = subword_embeddings[start:end]
            # Handle cases where a "word" might not have any subwords (e.g., empty strings).
            if subword_slice.shape[0] == 0:
                word_embeddings.append(torch.zeros_like(subword_embeddings[0]))
                continue
            # Combine the subword embeddings into a single vector using the chosen strategy.
            if self.pooling_strategy == 'mean':
                word_embedding = torch.mean(subword_slice, dim=0)

            elif self.pooling_strategy == 'sum':
                word_embedding = torch.sum(subword_slice, dim=0)

            elif self.pooling_strategy == 'attentive':
                # --- Attentive Pooling Logic ---
                # 1. Project the subword embeddings and apply a non-linear activation.
                projected_subwords = torch.tanh(self.attention_proj(subword_slice))
                
                # 2. Calculate an "importance score" for each subword.
                # Shape: [num_subwords_in_word, 1]
                scores = self.attention_scorer(projected_subwords)
                
                # 3. Use softmax to convert the scores into a probability distribution (weights).
                # Shape: [num_subwords_in_word, 1]
                attention_weights = F.softmax(scores, dim=0)
                
                # 4. Calculate the final word embedding as the weighted average of the subword embeddings.
                # This allows the model to focus on the most important subwords.
                word_embedding = torch.sum(attention_weights * subword_slice, dim=0)

            word_embeddings.append(word_embedding)
            
        # Stack the list of word embeddings into a single tensor.
        return torch.stack(word_embeddings, dim=0)