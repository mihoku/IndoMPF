import torch
import torch.nn as nn
import itertools
import os
import json
from tqdm import tqdm # A library for progress bars
import re

# --- PyTorch Geometric Libraries ---
# We need these to create the dataset and the dataloader.
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from transformers import BertTokenizer, AutoTokenizer, AutoModel

# --- Text Processing Libraries ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from subword_aggregation import SubwordAggregation
from plm_encoder import PLMEncoder

class ProgramDataset(Dataset):
    """
    A custom PyTorch Geometric Dataset to handle the JSON files.
    This class manages loading the data, processing it into graphs, and
    caching the results for efficiency.
    """
    def __init__(self, root, json_files, op_list, const_list, program_length, transform=None, pre_transform=None, tokenizer='deepseek_qwen', max_rows=1500, max_cols=1500):
        """
        Args:
            root (str): The root directory where the dataset should be stored.
                        This is used for caching processed data.
            json_files (list): A list of file paths to the raw JSON data.
            op_list, const_list: The program vocabulary.
            program_length (int): The fixed length to pad/truncate the answer programs to.
        """
        # self.processed_names = []
        self.json_files = [json_files]
        self.op_list = op_list
        self.const_list = const_list
        self.program_length = program_length
        self.vocab = self.op_list + self.const_list
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.tokenizer_name = tokenizer

        if tokenizer == 'deepseek_qwen':
            local_model_path = "deepseek.qwen"
            self.language_model = AutoModel.from_pretrained(local_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            self.sep = '<|endoftext|>' # The special separator token for Qwen-style models
            self.hidden_channels=1536

        elif tokenizer == 'gemma3':
            local_model_path = "gemma3.model"
            self.language_model = AutoModel.from_pretrained(local_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            self.sep = '<eos>' # The special separator token for Gemma-style models
            self.hidden_channels=1152

        elif tokenizer == 'gemma3.finetune':
            # language_model = AutoModel.from_pretrained('FacebookAI/xlm-roberta-base')
            # local_model_path = "gemma3.270m.finetune.embedding.indompf"
            local_model_path = "gemma3.embedding.finetune.mnr"
            self.language_model = AutoModel.from_pretrained(local_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            self.sep = '<eos>' # The special separator token for Gemma-style models
            self.hidden_channels = self.language_model.config.hidden_size

        elif tokenizer == 'xlm_roberta_base':
            local_model_path = "xlm.roberta.base"
            self.language_model = AutoModel.from_pretrained(local_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            self.sep = '<s>' # The special separator token for RoBERTa-style models
            self.hidden_channels=768

        elif tokenizer == 'xlm_roberta_large':
            local_model_path = "xlm.roberta.large"
            self.language_model = AutoModel.from_pretrained(local_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            self.sep = '<s>'
            self.hidden_channels=1024

        elif tokenizer == 'indobert_base_uncased':
            local_model_path = "indobert.base.uncased"
            self.language_model = AutoModel.from_pretrained(local_model_path)
            self.tokenizer = BertTokenizer.from_pretrained(local_model_path)
            self.sep = '[SEP]' # The special separator token for BERT-style models
            self.hidden_channels=768

        elif tokenizer == 'indobert_large_p2':
            self.language_model = AutoModel.from_pretrained('indobenchmark/indobert-large-p2')
            self.tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-large-p2')
            self.sep = '[SEP]'
            self.hidden_channels=1024
        
        self.max_rows = max_rows
        self.max_cols = max_cols
        
         # 1. Intialize PLM Encoder
        # self.plm_encoder = PLMEncoder(text_encoder_name)
        plm_hidden_size = self.language_model.config.hidden_size
        self.aggregation = SubwordAggregation(hidden_size=plm_hidden_size)
        
        # 2. Create the learnable embedding layers here ---
        # self.row_embedding = nn.Embedding(max_rows + 1, plm_hidden_size)
        # self.col_embedding = nn.Embedding(max_cols + 1, plm_hidden_size)

        # Initialize location embedding layers
        # moved to graph encoder
        # self.row_embedding = nn.Embedding(max_rows + 1, self.hidden_channels)
        # self.col_embedding = nn.Embedding(max_cols + 1, self.hidden_channels)
        
        super(ProgramDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        # This tells PyG where to find the raw source files.
        return [os.path.basename(f) for f in self.json_files]

    @property
    def processed_file_names(self):
        # This defines the names of the processed files. PyG will look for these
        # and skip the `process` step if they already exist.
        # We create one .pt file for each question-table pair.
        self.processed_names = []
        for f in self.json_files:
            all_data = json.load(open(f))
            for data in all_data:
                table_uid = data['table']['uid']
                for q in data['questions']:
                    self.processed_names.append(f"graph_{table_uid}_{q['uid']}_{self.tokenizer_name}.pt")
        return self.processed_names

    def _build_graph(self, full_json_data, table_uid, question_uid):
        """
        Constructs a PyG `Data` object with a WORD-LEVEL graph structure.
        Crucially, it also prepares the necessary subword information that will be
        needed by the pre-trained language model (PLM) and the subword aggregation layer.
        """
        # --- Step 1: Find and Extract Data ---
        # Locate the correct table, paragraphs, and question from the JSON data using their UIDs.
        table_data = full_json_data["table"]["table"]
        table_title = full_json_data["table"]["title"]
        paragraphs_data = full_json_data.get("paragraphs", [])
        # Prepend the table title to the paragraphs to include it in the context.
        paragraphs_data.insert(0, {"text": full_json_data.get("table", {}).get("title", "Table title not provided")})
        question_data = next((q for q in full_json_data.get("questions", []) if q.get("uid") == question_uid), None)
        if not question_data:
            return None
        question_text = question_data["question"]

        # --- Step 2: Define Relation Vocabulary ---
        # This dictionary maps human-readable relation names to unique integer IDs.
        # These IDs will become the `edge_type` in our graph.
        RELATION_VOCAB = {
            "question_to_next_word": 0, "paragraph_to_next_word": 1, "paragraph_to_paragraph_word":2,
            "table_cell_to_next_cell_in_row": 3, "table_cell_to_next_cell_in_col": 4,
            "table_cell_in_the_same_row": 5, "table_cell_in_the_same_col": 6,
            "table_cell_to_row_header": 7, "table_cell_to_col_header": 8,
            "question_to_paragraph_sentence": 9, "paragraph_sentence_to_question": 10,
            "question_to_paragraph_word": 11, "paragraph_word_to_question": 12,
            "question_to_table": 13, "table_to_question": 14,
            "paragraph_sentence_to_table": 15, "table_to_paragraph_sentence": 16,
            "paragraph_word_to_table": 17, "table_to_paragraph_word": 18
        }

        modality_vocab = {'table-number':0,'table-period':1, 'table-text':2, 'question':3, 'paragraph':4}

        # --- Step 3: Two-Stage Node and Subword Creation ---
        # This is the core of the word-level graph creation process.
        all_words = []                 # This will store the final word-level tokens, which are our nodes.
        all_subwords = []              # This will store the subword-level tokens for the PLM.
        word_to_subword_mapping = []   # Stores (start, end) indices to map each word to its subwords.
        
        # This list will store the (row, col) for each word.
        # It will be None for non-table words.
        node_to_location = []

        # This list will store the modality type of each node
        modality_type_list = []

        paragraph_and_word_node = {}
        
        def process_text(text, modality, location=None):
            """A helper function to perform two-stage tokenization on a piece of text."""
            
            is_first_word = not all_words
            # For table cells and paragraphs, the entire text is a single node.
            node_text = str(text)
            all_words.append(node_text)
            node_to_location.append(location)
            modality_type_list.append(modality)
            
            subwords = self.tokenizer.tokenize(node_text if is_first_word else " " + node_text)
            start_idx = len(all_subwords)
            all_subwords.extend(subwords)
            word_to_subword_mapping.append((start_idx, len(all_subwords)))

        def is_period(text: str) -> bool:
            """
            Checks if a string represents a time period.

            This function uses regular expressions to match several common date/time formats:
            - A standalone 4-digit year (e.g., "2023").
            - A month name followed by a year (e.g., "Nov 2023", "Januari 2024").
            - A compact year-month format (e.g., "202301" for January 2023).

            Args:
                text: The string to check.

            Returns:
                True if the string matches a period format, False otherwise.
            """
            if not isinstance(text, str):
                return False
                
            text = text.strip()

            # Pattern 1: A 4-digit year (e.g., 2023)
            # ^ and $ ensure the entire string must be a 4-digit year.
            year_pattern = r"^\d{4}$"
            if re.fullmatch(year_pattern, text):
                # Additional check to avoid matching generic 4-digit numbers
                try:
                    if 1900 <= int(text) <= 2100:
                        return True
                except ValueError:
                    pass # Not a valid integer

            # Pattern 2: Month and Year (e.g., "Nov 2023" or "November 2023")
            # This pattern looks for a month name, optional punctuation, whitespace, and a 4-digit year.
            months = [
                "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
                "january", "february", "march", "april", "june", "july", "august", "september",
                "oktober", "november", "desember", "januari", "maret", "mei", "juni", "juli", "agustus"
            ]
            month_regex = "|".join(months)
            month_year_pattern = rf"^(?:{month_regex})[\.,]?\s+\d{{4}}$"
            if re.fullmatch(month_year_pattern, text, re.IGNORECASE):
                return True
                
            # Pattern 3: Compact Year-Month (e.g., 202301)
            # Checks for a year (20xx) followed by a valid month (01-12).
            yyyymm_pattern = r"^20\d{2}(0[1-9]|1[0-2])$"
            if re.fullmatch(yyyymm_pattern, text):
                return True

            return False

        def is_number(text: str) -> bool:
            """
            Checks if a string can be robustly interpreted as a number.

            This function is designed to handle:
            - Integers and floating-point numbers.
            - Thousand separators, using either commas (,) or dots (.).
            - A single comma as a decimal separator (common in Indonesian format).

            Args:
                text: The string to check.

            Returns:
                True if the string can be converted to a float, False otherwise.
            """
            if not isinstance(text, str) or not text:
                return False

            # Standardize the number string for conversion.
            cleaned_text = text.strip()
            
            # 1. Remove all dots, as they are likely thousand separators in this context.
            cleaned_text = cleaned_text.replace('.', '')
            
            # 2. If a comma is present, assume it's the decimal separator and replace it with a dot.
            if ',' in cleaned_text:
                cleaned_text = cleaned_text.replace(',', '.', 1) # Replace only the first one
                # If there are still commas, it's not a valid number (e.g., "1,2,3")
                if ',' in cleaned_text:
                    return False

            # 3. Try to convert the cleaned string to a float.
            # This is the most reliable way to check if it's a number.
            try:
                float(cleaned_text)
                return True
            except ValueError:
                return False

        def identify_cell_type(cell_content: str) -> str:
            """
            Identifies the type of content in a table cell.

            The function checks in a specific order to avoid ambiguity:
            1. Is it a period (like a year)?
            2. If not, is it a number?
            3. If not, it must be text.

            Args:
                cell_content: The string content from the table cell.

            Returns:
                A string: "table-period", "table-number", or "table-text".
            """
            # Sanitize input by converting to string and stripping whitespace
            content_str = str(cell_content).strip()

            # If the cell is empty after stripping, classify it as text (or "empty" if you prefer).
            if not content_str:
                return "table-text"

            # The order of these checks is very important.
            # We must check for "period" first, because a year like "2023" is also a valid number.
            if is_period(content_str):
                return "table-period"
            
            if is_number(content_str):
                return "table-number"

            # If it's neither a period nor a number, it's classified as text.
            return "table-text"

        # Process all text sources to populate the master lists.
        question_words = str(question_text.replace("?","")).split()
        for word in question_words:
            process_text(word, modality=modality_vocab["question"])
        # process_text(question_text.replace("?",""), is_first_word_global=True)
        # Add a separator token after the question ---
        # process_text(self.sep, is_first_word_global=False)
        
        paragraph_start_idx = len(all_words)
        for p in paragraphs_data:
            paragraph_text = p.get("text", "")
            # to create a single node for paragraph
            process_text(paragraph_text, modality=modality_vocab["paragraph"])
            paragraph_node_idx = len(all_words)
            for paragraph_word in paragraph_text.split():
                # to create a single node for each word in paragraph
                process_text(paragraph_word, modality=modality_vocab["paragraph"])
            paragraph_word_node_start_idx = paragraph_node_idx+1
            paragraph_word_node_end_idx = len(all_words)
            paragraph_and_word_node[paragraph_node_idx] = (paragraph_word_node_start_idx, paragraph_word_node_end_idx)

        table_start_idx = len(all_words)

        # Add a separator token after the paragraphs ---
        # process_text(self.sep, is_first_word_global=False)
        
        # This dictionary maps a cell's (row, col) coordinate to its range of WORD indices in `all_words`.
        cell_word_indices = {}
        for r, row in enumerate(table_data):
            for c, cell in enumerate(row):
                table_cell_idx = len(all_words)
                modality_type = modality_vocab[identify_cell_type(cell)]
                process_text(str(cell) if str(cell) else "[EMPTY]", modality = modality_type, location=(r, c))
                cell_word_indices[(r, c)] = table_cell_idx

        
        # --- Step 4: Edge Creation (at the WORD level) ---
        # Now we build the graph's connections using the word-level indices.
        src_edges, dst_edges, relation_types = [], [], []
        def add_edge(src, dst, rel_name):
            src_edges.append(src)
            dst_edges.append(dst)
            relation_types.append(RELATION_VOCAB[rel_name])
        
        # The edge creation logic connects nodes based on their structural relationships.
        question_len = paragraph_start_idx
        paragraph_len = table_start_idx - paragraph_start_idx
        # Connect adjacent words in the question
        for i in range(question_len - 1):
            add_edge(i, i + 1, "question_to_next_word"); add_edge(i + 1, i, "question_to_next_word")
        # Connect paragraph to words and adjacent words in the paragraphs
        for paragraph_node, paragraph_word_nodes_range in paragraph_and_word_node.items():
            range_start, range_end = paragraph_word_nodes_range
            for i in range(range_start, range_end):
                add_edge(paragraph_node, i, "paragraph_to_paragraph_word")
                if i+1!=range_end:
                    src, dst = i, i + 1
                    add_edge(src, dst, "paragraph_to_next_word")

        # Connect table cells based on row, column, and header relationships
        for r, row in enumerate(table_data):
            for c, cell in enumerate(row):
                # We connect the first word of each cell to represent cell-level connections
                cell_node_idx = cell_word_indices.get((r,c))

                # connect current cell to all cells in the same column 
                for i in range(1,len(table_data)):
                    if i!=c:
                        cell_in_same_col_idx = cell_word_indices[(i, c)]
                        add_edge(cell_node_idx, cell_in_same_col_idx, "table_cell_in_the_same_col")

                # connect current cell to all cells in the same row
                for i in range(1,len(row)):
                    if i!=r:
                        cell_in_same_row_idx = cell_word_indices[(r, i)]
                        add_edge(cell_node_idx, cell_in_same_row_idx, "table_cell_in_the_same_row")

                if c + 1 < len(row): 
                    next_cell_col_idx = cell_word_indices[(r, c + 1)]
                    add_edge(cell_node_idx, next_cell_col_idx, "table_cell_to_next_cell_in_row")

                if r + 1 < len(table_data): 
                    next_cell_row_idx = cell_word_indices[(r + 1, c)]
                    add_edge(cell_node_idx, next_cell_row_idx, "table_cell_to_next_cell_in_col")
                
                if r > 0: 
                    header_start_idx = cell_word_indices[(0, c)]
                    add_edge(cell_node_idx, header_start_idx, "table_cell_to_col_header")
                
                if c > 0: 
                    header_start_idx = cell_word_indices[(r, 0)]
                    add_edge(cell_node_idx, header_start_idx, "table_cell_to_row_header")
        
        # --- Part B: Inter-context Edges (Sparse, Meaning-based) ---
        # Create a dense web of connections between the different contexts (question, paragraph, table).

        # Define stopwords for English and Bahasa Indonesia
        stopwords_en = {'a', 'an', 'the', 'in', 'on', 'at', 'for', 'to', 'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'can', 'could', 'not', 'and', 'or', 'but', 'if', 'because', 'as', 'until', 'while', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'just', 'don', 'now'}
        stopwords_id = {'yang', 'di', 'dan', 'dari', 'ke', 'dengan', 'ini', 'itu', 'adalah', 'untuk', 'pada', 'oleh', 'sebagai', 'bahwa', 'dalam', 'saya', 'kami', 'anda', 'dia', 'mereka', 'kita', 'bisa', 'akan', 'sudah', 'belum', 'tidak', 'bukan', 'ada', 'saja', 'juga', 'harus', 'jangan', 'telah', 'saat', 'ketika', 'mana', 'siapa', 'apa', 'mengapa', 'bagaimana', 'karena', 'agar', 'supaya', 'atau', 'tapi', 'namun', 'serta', 'yaitu', 'yakni'}
        all_stopwords = stopwords_en.union(stopwords_id)

        # 1. Connect Question and Paragraphs based on exact word matches
        q_indices = range(question_len)
        #p_indices = range(paragraph_start_idx, table_start_idx)
        for q_idx in q_indices:
            q_word = all_words[q_idx]
            if q_word in all_stopwords:
                continue
            
            for paragraph_node, paragraph_word_nodes_range in paragraph_and_word_node.items():
                # if paragraph sentence contains question word
                if q_word in all_words[paragraph_node]:
                    add_edge(q_idx, paragraph_node, "question_to_paragraph_sentence")
                    add_edge(q_idx, paragraph_node, "paragraph_sentence_to_question")
                
                range_start, range_end = paragraph_word_nodes_range
                for p_word_idx in range(range_start,range_end):
                    if q_word==all_words[p_word_idx]:
                        add_edge(q_idx, p_word_idx, "question_to_paragraph_word")
                        add_edge(p_word_idx, q_idx, "paragraph_word_to_question")                    

        
        # 2. Connect Question/Paragraphs to Table
        for r, row in enumerate(table_data):
            for c, cell in enumerate(row):
                # We connect the first word of each cell to represent cell-level connections
                cell_node_idx = cell_word_indices.get((r,c))

                # Check against question words
                for q_idx in q_indices:
                    q_word = all_words[q_idx]
                    if q_word in all_stopwords: continue
                    if q_word in all_words[cell_node_idx]:
                        add_edge(q_idx, cell_node_idx, "question_to_table")
                        add_edge(cell_node_idx, q_idx, "table_to_question")

                # Check against paragraph
                for paragraph_node, paragraph_word_nodes_range in paragraph_and_word_node.items():
                    # if paragraph sentence contains question word
                    if all_words[cell_node_idx] in all_words[paragraph_node]:
                        add_edge(cell_node_idx, paragraph_node, "table_to_paragraph_sentence")
                        add_edge(cell_node_idx, paragraph_node, "paragraph_sentence_to_table")
                    
                    range_start, range_end = paragraph_word_nodes_range
                    for p_word_idx in range(range_start,range_end):
                        if all_words[p_word_idx] in all_words[cell_node_idx]:
                            add_edge(cell_node_idx, p_word_idx, "table_to_paragraph_word")
                            add_edge(p_word_idx, cell_node_idx, "paragraph_word_to_table")

        # --- Step 5: Create PyTorch Geometric Data Object ---
        # Bundle all the processed information into a single `Data` object.
        edge_index = torch.tensor([src_edges, dst_edges], dtype=torch.long)
        edge_type = torch.tensor(relation_types, dtype=torch.long)
        
        # all_words_in_batch = [word for graph_tokens in all_words for word in graph_tokens]
        
        # # Aggregate subword embeddings to get word-level node features.
        # word_level_features_list = []
        # with torch.no_grad():
        #     for word in all_words:
        #         inputs = self.tokenizer(f' {word}', return_tensors='pt')
        #         outputs = self.language_model(**inputs)
        #         # Average subword tokens (ignoring <s> and </s>)
        #         embedding = outputs.last_hidden_state[0, 1:-1].mean(dim=0)
        #         word_level_features_list.append(embedding)

        # Stack the collected embeddings into a single tensor for the GNN
        # word_features_tensor = torch.stack(word_level_features_list)

        # B. Get location embeddings
        null_row_idx = self.max_rows
        null_col_idx = self.max_cols
        row_indices = torch.tensor([loc[0] if loc else null_row_idx for loc in node_to_location])
        col_indices = torch.tensor([loc[1] if loc else null_col_idx for loc in node_to_location])
        # r_embed = self.row_embedding(row_indices)
        # c_embed = self.col_embedding(col_indices)
        modality_tensor = torch.tensor(modality_type_list)

        subword_ids=torch.tensor(self.tokenizer.convert_tokens_to_ids(all_subwords), dtype=torch.long)

        # --- Step 1: Get subword embeddings from the FROZEN PLM ---
        with torch.no_grad():
            # This call runs without tracking gradients, saving memory and compute.
            # The PLM's weights will not be updated.
            # subword_embeddings = self.plm_encoder(subword_ids)
            subword_embeddings = self.language_model(subword_ids.unsqueeze(0)).last_hidden_state.squeeze(0)
        
            # --- Step 2: Aggregate subword embeddings (Trainable) ---
            # This part is outside the no_grad block, so if your aggregation layer
            # had learnable parameters, they would be trained.
            word_level_features = self.aggregation(subword_embeddings, word_to_subword_mapping)
            
            node_level_features = word_level_features

        graph_data = Data(
            num_nodes=len(all_words), # The number of nodes is the number of words.
            #  edge_index is attribute for defining the graph's structure. 
            # It's a 2D tensor of shape [2, num_edges] that stores the connections (edges) 
            # in what's called a Coordinate (COO) format.
            # The first row is a list of source nodes. 
            # The second row is a list of target nodes.
            edge_index=edge_index,
            # edge_type tensor holds the edge features. It's a 2D tensor of shape [num_edges, num_edge_features]. 
            # Each row corresponds to an edge defined in edge_index and describes its properties.
            edge_type=edge_type,
            tokens=all_words, # The node identifiers are the actual word strings.
            # location embeddings
            row_indices = row_indices,
            col_indices = col_indices,
            #x=word_features_tensor, # The 'x' attribute is standard for node features
            # x = final_x, # The 'x' attribute is standard for node features and location embedding
            x = node_level_features,
            # These two new attributes are crucial for the model's forward pass.
            #subword_ids=torch.tensor(self.tokenizer.convert_tokens_to_ids(all_subwords), dtype=torch.long),
            #word_to_subword_mapping=torch.tensor(word_to_subword_mapping, dtype=torch.long),
            node_locations=node_to_location,
            modality_list = modality_tensor,
            table_uid=table_uid,
            question_uid=question_uid 
        )
        return graph_data

    def process(self):
        """
        This is the core method of the dataset. It runs ONCE.
        It iterates through all raw JSON files, processes them into graph `Data` objects,
        and saves them to the `self.processed_dir` directory.
        """
        idx = 0
        # self.processed_names = []
        for file_path in tqdm(self.raw_paths, desc="Processing dataset"):
            all_data = json.load(open(file_path))
            for data in tqdm(all_data, desc="Creating Tabular Graph"):
                table_uid = data['table']['uid']
                
                for question_data in data['questions']:

                    # continue if the question is not valid
                    if question_data["validity_status"]!="matched facts (program parseable)":
                         self.processed_names.remove(f'graph_{table_uid}_{question_uid}_{self.tokenizer_name}.pt')
                         continue
                    else:
                        question_uid = question_data['uid']
                        
                        if os.path.exists(os.path.join(self.processed_dir, f'graph_{table_uid}_{question_uid}_{self.tokenizer_name}.pt')):
                            # print("Tabular graph has been created.")
                            continue

                        # 1. Build the graph for the current table-question pair.
                        #try:
                        graph_data = self._build_graph(data, table_uid, question_uid)
                        # except Exception as e:
                        #     print(f"Graph building failed for graph_{table_uid}_{question_uid}_{self.tokenizer_name}.pt")
                        #     print(f"{e}")
                        #     self.processed_names.remove(f'graph_{table_uid}_{question_uid}_{self.tokenizer_name}.pt')
                        #     break
                        
                        if graph_data is None:
                            print(f"Graph is empty for graph_{table_uid}_{question_uid}_{self.tokenizer_name}.pt")
                            self.processed_names.remove(f'graph_{table_uid}_{question_uid}_{self.tokenizer_name}.pt')
                            continue
                        
                        # 2. Tokenize the answer program.
                        # The 'reasoning' block contains the sequence of operations.
                        try:
                            program_tokens = self._extract_program_from_reasoning(question_data['reasoning'])
                        # if error
                        except:
                            print(f"Graph building failed for graph_{table_uid}_{question_uid}_{self.tokenizer_name}.pt")
                            self.processed_names.remove(f'graph_{table_uid}_{question_uid}_{self.tokenizer_name}.pt')
                            continue

                        ##3. Tokenize the program using the pointing mechanism.
                        #    This method needs info from the graph build step.
                        # program_ids = self._tokenize_program(program_tokens)
                        program_ids = self._tokenize_program_with_pointing(
                            program_tokens,
                            graph_data.node_locations,
                            graph_data.tokens
                        )
                        
                        # 3. Pad/truncate the program to a fixed length.
                        padded_program_ids = self._pad_program(program_ids)
                        
                        # 4. Attach the tokenized program as the label 'y' to the graph data object.
                        graph_data.y = padded_program_ids
                        
                        # Save the processed `Data` object to a file.
                        torch.save(graph_data, os.path.join(self.processed_dir, f'graph_{table_uid}_{question_uid}_{self.tokenizer_name}.pt'))
                        # self.processed_names.append(f'graph_{table_uid}_{question_uid}.pt')
                        print(f'graph_{table_uid}_{question_uid}_{self.tokenizer_name}.pt successfully generated!')
                        idx += 1
        
        filename = f"processed_tabular_graph_{self.tokenizer_name}.txt"
        with open(filename, 'w') as file_handler:
            # Loop through each item in the list
            for item in self.processed_names:
                # Write the item to the file, followed by a newline character
                file_handler.write(f"{item}\n")
        
        print("Pre-computation complete. Freeing PLM from memory.")
        del self.language_model
        del self.tokenizer
        torch.cuda.empty_cache()
    
    def _extract_program_from_reasoning(self, reasoning_dict):
        """Helper to convert the reasoning JSON into a flat list of program tokens."""

        tokens = ['GO']
        # The reasoning steps must be sorted by key (e.g., #0, #1, #2...)
        sorted_steps = sorted(reasoning_dict.items(), key=lambda item: int(item[0][1:]))
        
        for key, step in sorted_steps:
            tokens.append(step['operation'])
            tokens.append('(')
            for op in step['operands']:
                if isinstance(op, str):
                    tokens.append(op)
                elif isinstance(op, dict):
                    if op.get('source') == 'table':
                            # This ensures both elements are integers before creating the tuple
                            tokens.append(tuple(int(x) for x in op['location']))
                    elif op.get('source') == 'constant':
                        tokens.append(f"const_{op['fact']}")
                    elif '#' in op.get('source', ''):
                        tokens.append(op['source'])
                    else:
                        raise ValueError(f"Unknown operand source: {op.get('source')}")
                tokens.append(',')
            if tokens[-1] == ',':
                tokens.pop()
            tokens.append(')')
        tokens.append('EOS')
        return tokens

    # def _tokenize_program(self, tokens):
    #     """Converts a list of token strings into a list of integer IDs."""
    #     # A real implementation would need to handle pointing to table values.
    #     # This simplified version assumes all operands are in the vocab.
    #     return [self.token_to_id.get(token, 0) for token in tokens] # Default to 0 for unknown tokens

    def _tokenize_program_with_pointing(self, tokens, node_to_location, all_words):
        """
        Converts a list of token strings into a list of integer IDs.
        Implements a pointing mechanism for table cell operands.
        """
        program_ids = []
        vocab_size = len(self.token_to_id)
        
        # Create a reverse map from (row, col) to the first word node index
        # This is a bit slow, but robust. It runs only once per graph.
        location_to_node_idx = {loc: i for i, loc in enumerate(node_to_location) if loc is not None}

        for token in tokens:
            if isinstance(token, str):
                # It's a regular token (operation, constant, '(', ')', etc.)
                # Default to 0 (padding ID) for unknown tokens
                program_ids.append(self.token_to_id.get(token, 0))
            
            elif isinstance(token, tuple):
                # This is a location tuple, e.g., (2, 3), for a table operand.
                # We need to find the corresponding node index.
                
                # Find the first node index that matches this location
                # node_idx = -1
                # for i, loc in enumerate(node_to_location):
                #     if loc == token:
                #         node_idx = i
                #         break
                # Use the dictionary for a fast, O(1) lookup.
                node_idx = location_to_node_idx.get(token, -1)
                
                if node_idx != -1:
                    # Pointing mechanism: ID is vocab_size + node_index
                    program_ids.append(vocab_size + node_idx)
                else:
                    # Fallback: couldn't find the location in the graph.
                    # This might happen if the cell was empty. Default to padding.
                    print(f"Warning: Could not find location {token} in the graph. Defaulting to pad token.")
                    program_ids.append(0)
            else:
                # Should not happen
                program_ids.append(0)

        return program_ids

    def _pad_program(self, ids):
        """Pads or truncates a list of token IDs to `self.program_length`."""
        padded = ids[:self.program_length]
        padded.extend([0] * (self.program_length - len(padded))) # 0 is the padding ID
        return torch.tensor(padded, dtype=torch.long)

    def len(self):
        # Returns the total number of processed graph files.
        return len(self.processed_file_names)

    def get(self, idx):
        # This method is called by the DataLoader. It loads a single processed
        # graph file from disk.
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]),weights_only=False)
        return data