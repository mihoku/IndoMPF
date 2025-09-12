import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch


# --- The Transformer Decoder ---
class TransformerProgramDecoder(nn.Module):
    """
    A Transformer-based decoder for generating the program sequence.
    It is designed to handle a hybrid input of fixed vocabulary tokens
    and pointers to the encoder's output nodes.
    """
    def __init__(self, hidden_size, program_length, op_list, const_list, max_input_length,
                 num_layers=4, num_heads=4, dropout_rate=0.1):
        super(TransformerProgramDecoder, self).__init__()

        # --- Vocabulary and Dimension Configuration ---
        self.op_list = op_list
        self.const_list = const_list
        self.op_list_size = len(op_list)
        self.const_list_size = len(const_list)
        self.reserved_token_size = self.op_list_size + self.const_list_size
        self.program_length = program_length
        self.hidden_size = hidden_size
        self.max_input_length = max_input_length
        self.full_vocab_size = self.reserved_token_size + self.max_input_length

        # --- Pre-computed Indices for Special Tokens ---
        self.go_token_id = self.op_list.index('GO')
        self.eos_token_id = self.op_list.index('EOS')
        self.open_paren_id = self.op_list.index('(')
        self.close_paren_id = self.op_list.index(')')
        self.comma_id = self.op_list.index(',')
        
        # --- Pre-computed Indices for Operands and Operators ---
        self.op_indices = [i for i, token in enumerate(self.op_list) if token not in ['GO', 'EOS', '(', ')', ','] and '#' not in token]
        self.input_node_indices = range(self.reserved_token_size, self.full_vocab_size)
        self.step_token_indices = [i for i, token in enumerate(self.op_list) if '#' in token]
        self.constant_token_indices = range(self.op_list_size, self.reserved_token_size)

        # Pre-computed Indices for Operators based on Special Characteristics
        one_operand = ["retrieve"]
        self.operator_one_operand = [i for i, token in enumerate(self.op_list) if token in one_operand]
        two_operands = ["addition", "subtraction", "multiplication", "division", "exponential", "greater", "smaller", "equal"]
        self.operator_two_operands = [i for i, token in enumerate(self.op_list) if token in two_operands]
        first_operand_constant = ["maximum_n", "minimum_n", "trace_column", "trace_row"]
        self.operator_first_operand_constant = [i for i, token in enumerate(self.op_list) if token in first_operand_constant]
        conditional_operator = ["count_if_equal", "count_if_less", "count_if_greater", "count_if_geq", "count_if_leq", "count_if_notequal", "sum_if_equal", "sum_if_less", "sum_if_greater", "sum_if_geq", 
                                "sum_if_leq", "sum_if_notequal", "filter_if_equal", "filter_if_less", "filter_if_greater", "filter_if_geq", "filter_if_leq", "filter_if_notequal"]
        self.operator_two_or_more_operand = [i for i, token in enumerate(self.op_list) if token in first_operand_constant+conditional_operator]
        require_number_modality = ["addition", "subtraction", "multiplication", "division", 
                                    "exponential", "greater", "smaller", "equal", "sum", "average", 
                                    "minimum", "maximum", "maximum_n", "minimum_n"]
        self.operator_number_modality = [i for i, token in enumerate(self.op_list) if token in require_number_modality]

        # --- Core Model Layers ---
        # 1. Token Embedding: This layer ONLY handles the fixed vocabulary.
        #    The features for pointer tokens come directly from the encoder.
        self.token_embedding = nn.Embedding(self.reserved_token_size, hidden_size)
        
        # 2. Positional Embedding
        self.positional_embedding = nn.Parameter(torch.randn(1, program_length, hidden_size))
        
        # 3. The Transformer Decoder Stack
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.decoder_layers = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # --- Grammar-based Decoding Masks ---
        self._create_grammar_masks()
        self.dropout = nn.Dropout(dropout_rate)

    def _pad_program(self, ids):
        """Pads or truncates a list of token IDs to `self.program_length`."""
        padded = ids[:self.program_length]
        padded.extend([0] * (self.program_length - len(padded))) # 0 is the padding ID
        return torch.tensor(padded, dtype=torch.long)

    def _create_grammar_masks(self):
        """
        Helper function to create the boolean masks needed for constrained decoding.
        Its job is to create a set of "stencils" or "templates" that enforce a simple grammar, 
        ensuring the generated program is always syntactically correct 
        (e.g., an operator is always followed by an open parenthesis). 
        It does this by creating boolean tensors where True means "this token is allowed" 
        and False means "this token is forbidden."
        """
        
        # The total size of the output vocabulary, 
        # including fixed tokens and all possible pointer slots.
        full_space_size = self.full_vocab_size
        
        # Identify the indices of all valid operators. This excludes special tokens like 'GO', 'EOS',
        # parentheses, commas, and step references (like '#0', '#1').
        # Purpose: This mask is used when the decoder is in the expect_op state 
        #           (e.g., at the beginning of the program or after a closing parenthesis).
        # Result: Only allows the model to choose a valid function like sum, filter, count, etc.
        # Create a mask that is initially all False.
        mask_op = torch.zeros(full_space_size, dtype=torch.bool)
        # Set the positions for valid operators to True.
        mask_op[self.op_indices] = True

        # Identify all graph tokens
        mask_graph_tokens_only = torch.zeros(full_space_size, dtype=torch.bool)
        mask_graph_tokens_only[self.input_node_indices] = True
        
        # Create a mask that is initially all False.
        # Purpose: Used in the expect_open_paren state, right after an operator has been generated.
        # Result: Forces the model to generate a (.
        mask_open_paren = torch.zeros(full_space_size, dtype=torch.bool)
        # Allow only the open parenthesis token.
        mask_open_paren[self.open_paren_id] = True

        # Create a mask that is initially all False.
        # Purpose: Used in the expect_comma_or_close state, right after an operator has been generated.
        # Result: Forces the model to generate a ).
        close_paren_mask = torch.zeros(full_space_size, dtype=torch.bool)
        # Allow only the open parenthesis token.
        close_paren_mask[self.close_paren_id] = True

        # Create a mask that is initially all False.
        # Purpose: Used in the expect_comma_or_close state, right after an operator has been generated.
        # But specifically, it is implemented in operator that requires more than one operand.
        # Result: Forces the model to generate a comma (,).
        comma_mask = torch.zeros(full_space_size, dtype=torch.bool)
        # Allow only the open parenthesis token.
        comma_mask[self.comma_id] = True
        
        # Create a mask that is initially all False.
        # Purpose: Used in the expect_arg state, after an open parenthesis or a comma.
        # Result: Allows the model to choose from three types of valid arguments 
        #           i.e. a predefined constant, step memory token (e.g. #1, #2),
        #               a pointer to an input node, or a closing parenthesis to end the function call.
        mask_arg = torch.zeros(full_space_size, dtype=torch.bool)
        # --- Allowable Arguments ---
        # 1. Allow any token from the predefined constant list (e.g., '1', '2','100').
        const_indices = range(self.op_list_size, self.reserved_token_size)
        mask_arg[const_indices] = True
        # 2. Allow the model to point to any node in the input graph.
        mask_arg[self.input_node_indices] = True
        # 3. Allow any step memory token (e.g., '#0', '#1', '#2').
        mask_arg[self.step_token_indices] = True
        
        # Create a mask that is initially all False.
        # Purpose: Used in the expect_comma_or_close state, after an argument has been generated.
        # Result: Forces the model to either add another argument (by generating ,) 
        #           or finish the function call (by generating )).
        mask_comma_or_close = torch.zeros(full_space_size, dtype=torch.bool)
        # 1. Allow the comma token, to add another argument.
        mask_comma_or_close[self.comma_id] = True
        # 2. Allow the closing parenthesis token, to finish the function call.
        mask_comma_or_close[self.close_paren_id] = True

        # Create a mask to enable EOS token or operator token
        # The inclusion of EOS token allows program generation to end after finishing a complete operator.
        op_or_eos_mask = mask_op.clone()
        op_or_eos_mask[self.eos_token_id] = True

        # Create a mask for constants
        mask_constants = torch.zeros(self.full_vocab_size, dtype=torch.bool)
        mask_constants[self.constant_token_indices] = True

        # Save these masks as part of the model's state.
        # 'register_buffer' ensures they are moved to the correct device (CPU/GPU) with the model,
        # but they are not considered model parameters and are not updated during training.
        self.register_buffer('op_mask', mask_op)
        self.register_buffer('open_paren_mask', mask_open_paren)
        self.register_buffer('arg_mask', mask_arg)
        self.register_buffer('comma_or_close_mask', mask_comma_or_close)
        self.register_buffer('op_or_eos_mask', op_or_eos_mask)
        self.register_buffer('constant_only_mask', mask_constants)
        self.register_buffer('close_paren_mask', close_paren_mask)
        self.register_buffer('comma_mask', comma_mask)
        self.register_buffer('mask_graph_tokens_only', mask_graph_tokens_only)

    def forward(self, encoder_output, input_mask, target_program):
        """
        The forward pass for TRAINING. It correctly handles a mix of vocabulary
        tokens and pointer tokens from the target_program sequence.
        """
        batch_size, seq_len = target_program.shape
        device = encoder_output.device

        # --- Step 1: Create Embeddings for the Hybrid Target Program ---
        
        # Create masks to identify which tokens are from the fixed vocab and which are pointers.
        vocab_mask = target_program < self.reserved_token_size
        pointer_mask = ~vocab_mask

        # Initialize the final embedded tensor that will be fed to the decoder layers.
        target_embedded = torch.zeros(batch_size, seq_len, self.hidden_size, device=device)

        # --- Part A: Handle Fixed Vocabulary Tokens ---
        if vocab_mask.any():
            # Get the IDs of the fixed vocab tokens, setting pointer IDs to 0 to avoid index errors.
            vocab_ids = target_program.masked_fill(pointer_mask, 0)
            # Get their embeddings from the dedicated embedding layer.
            vocab_embeddings = self.token_embedding(vocab_ids)
            # Place these embeddings in the correct positions in the final tensor using the mask.
            target_embedded[vocab_mask] = vocab_embeddings[vocab_mask]

        # --- Part B: Handle Pointer Tokens ---
        if pointer_mask.any():
            # Get the node indices by subtracting the vocab size.
            # Set fixed vocab IDs to 0 to avoid negative indices.
            pointer_node_indices = (target_program - self.reserved_token_size).masked_fill(vocab_mask, 0)
            
            # Expand indices to gather the full hidden state for each node.
            # Shape goes from [B, T] -> [B, T, H]
            expanded_indices = pointer_node_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
            
            # Use `gather` to select the corresponding node features from the encoder's output.
            # `encoder_output` has shape [B, N, H] where N is the number of nodes.
            # `expanded_indices` has shape [B, T, H] where T is the sequence length.
            pointer_features = torch.gather(encoder_output, 1, expanded_indices)

            # Place the gathered features into the correct positions using the mask.
            target_embedded[pointer_mask] = pointer_features[pointer_mask]

        # --- Step 2: Proceed with the Standard Transformer Decoder Logic ---
        causal_mask = self.generate_square_subsequent_mask(seq_len).to(device)
        
        # Add positional information.
        target_embedded = self.dropout(target_embedded + self.positional_embedding[:, :seq_len])
        
        # Pass the prepared inputs through the entire stack of Transformer decoder layers.
        decoder_output = self.decoder_layers(
            tgt=target_embedded,
            memory=encoder_output,
            tgt_mask=causal_mask,
            memory_key_padding_mask=~input_mask # PyTorch expects False for valid tokens
        )
        
        return decoder_output

    def get_all_masks(self):
        """
        Bundles all the pre-computed grammar masks and special token IDs into a
        single dictionary to be passed to the state machine.
        """
        return {
            'op_mask': self.op_mask,
            'open_paren_mask': self.open_paren_mask,
            'arg_mask': self.arg_mask,
            'comma_or_close_mask': self.comma_or_close_mask,
            'op_or_eos_mask': self.op_or_eos_mask,
            'constant_only_mask': self.constant_only_mask,
            'close_paren_mask': self.close_paren_mask,
            'comma_mask': self.comma_mask,
            'mask_graph_tokens_only': self.mask_graph_tokens_only,
            'eos_token_id': self.eos_token_id,
            'comma_id': self.comma_id,
            'close_paren_id': self.close_paren_id
        }

    def get_operator_rules(self):
        """
        Bundles all the pre-computed operator type lists into a single
        dictionary to be passed to the state machine.
        """
        return {
            'one_operand': self.operator_one_operand,
            'two_operands': self.operator_two_operands,
            'first_operand_constant': self.operator_first_operand_constant,
            'two_or_more_operand': self.operator_two_or_more_operand,
            'number_modality': self.operator_number_modality
        }

    @staticmethod
    def generate_square_subsequent_mask(sz):
        """Generates a square causal mask for a sequence of size sz."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
