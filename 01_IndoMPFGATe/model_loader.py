import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from dataset_processor import ProgramDataset
import os
import json
from graph_encoder import GraphEncoder
from decoder_transformer import TransformerProgramDecoder
from model import IndoMPFGATe
from tqdm import tqdm
import argparse
import datetime
from transformers import AutoModel
from sklearn.metrics import precision_recall_fscore_support, accuracy_score 
from torch.optim.lr_scheduler import ReduceLROnPlateau
import glob

exp_start = datetime.datetime.now()
timestamp_exp_start = exp_start.strftime("%Y%m%d%H%M%S")

BATCH_SIZE = 2
HIDDEN_DIM = 768
PROG_LEN = 1500
MAX_NODES = 150000
NUM_RELATIONS = 19


def train(args):
    """
    The main function to handle the model training process.
    """
    # --- 1. Setup ---
    print("--- Starting Training Setup ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cuda')
    print(f"Using device: {device}")

    # experiment id
    exp_id = timestamp_exp_start+"_"+args.exp_id if timestamp_exp_start!=args.exp_id else args.exp_id

    # Define the program vocabulary.
    op_list = ['GO', 'EOS', '(', ')', ',', 
               "retrieve", "addition", "subtraction", "multiplication", "division", 
                "exponential", "greater", "smaller", "equal", "sum", "average", 
                "minimum", "maximum", "maximum_n", "minimum_n", "count", "count_if_equal", 
                "count_if_less", "count_if_greater", "count_if_geq", "count_if_leq", 
                "count_if_notequal", "sum_if_equal", "sum_if_less", "sum_if_greater", "sum_if_geq", 
                "sum_if_leq", "sum_if_notequal", "filter_if_equal", "filter_if_less",
                "filter_if_greater", "filter_if_geq", "filter_if_leq", "filter_if_notequal",
                "trace_column", "trace_row", "unique"] + [f'#{i}' for i in range(1000)]
    const_list = [f'const_{i}' for i in range(1000)]
    full_vocab = op_list + const_list
    reserved_token_size = len(full_vocab)

    # --- 2. Data Loading ---
    
    # Create PyTorch Geometric Dataset objects.
    train_dataset = ProgramDataset(
        root=os.path.join(args.output_dir, 'train'),
        json_files=args.train_file,
        op_list=op_list, const_list=const_list, program_length=args.program_length,
        tokenizer=args.text_encoder
    )
    valid_dataset = ProgramDataset(
        root=os.path.join(args.output_dir, 'valid'),
        json_files=args.valid_file,
        op_list=op_list, const_list=const_list, program_length=args.program_length,
        tokenizer=args.text_encoder
    )
    test_dataset = ProgramDataset(
        root=os.path.join(args.output_dir, 'test'),
        json_files=args.test_file,
        op_list=op_list, const_list=const_list, program_length=args.program_length,
        tokenizer=args.text_encoder
    )

    vocab_size = train_dataset.tokenizer.vocab_size

    # Create DataLoaders to handle batching and shuffling.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=3)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,num_workers=3)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    print("DataLoaders created successfully.")

    if args.text_encoder == 'deepseek_qwen':
        # language_model = AutoModel.from_pretrained('FacebookAI/xlm-roberta-base')
        local_model_path = "deepseek.qwen"
        language_model = AutoModel.from_pretrained(local_model_path)
    elif args.text_encoder == 'gemma3':
        # language_model = AutoModel.from_pretrained('FacebookAI/xlm-roberta-base')
        local_model_path = "gemma3.model"
        language_model = AutoModel.from_pretrained(local_model_path)
    elif args.text_encoder == 'gemma3.finetune':
        # language_model = AutoModel.from_pretrained('FacebookAI/xlm-roberta-base')
        # local_model_path = "gemma3.270m.finetune.embedding.indompf"
        local_model_path = "gemma3.embedding.finetune.mnr"
        language_model = AutoModel.from_pretrained(local_model_path)
        embedding_dimension = language_model.config.hidden_size
        del language_model
    elif args.text_encoder == 'xlm_roberta_base':
        # language_model = AutoModel.from_pretrained('FacebookAI/xlm-roberta-base')
        local_model_path = "xlm.roberta.base"
        language_model = AutoModel.from_pretrained(local_model_path)
    elif args.text_encoder == 'xlm_roberta_large':
        local_model_path = "xlm.roberta.large"
        language_model = AutoModel.from_pretrained(local_model_path)
    elif args.text_encoder == 'indobert_base_uncased':
        local_model_path = "indobert.base.uncased"
        language_model = AutoModel.from_pretrained(local_model_path)
    elif args.text_encoder == 'indobert_large_p2':
        language_model = AutoModel.from_pretrained('indobenchmark/indobert-large-p2')

    # Access the configuration and retrieve the hidden_size
    

    # --- 3. Model, Loss, and Optimizer ---
    # Instantiate the full encoder-decoder model.
    encoder = GraphEncoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dimension,
        text_encoder_name=args.text_encoder,
        hidden_channels=args.HIDDEN_DIM,
        num_layers=args.encoder_layer,
        num_heads=args.encoder_head,
        num_relations=NUM_RELATIONS
    )
    decoder = TransformerProgramDecoder(
        hidden_size=args.HIDDEN_DIM, program_length=args.program_length, op_list=op_list, const_list=const_list,
        max_input_length=MAX_NODES, # Use the max nodes from the padded batch
        num_layers = args.decoder_layer,
        num_heads = args.decoder_head,
    )
    model = IndoMPFGATe(encoder, decoder).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    print("\nFull Transformer-based GraphProgramGenerator model instantiated successfully.")    
    # model = GraphProgramGenerator(None, None) # Placeholder
    
    # The loss function: CrossEntropyLoss is ideal for this token classification task.
    # We tell it to ignore the padding token (ID 0) when calculating the loss.
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # The optimizer: AdamW is a robust choice for Transformer-based models.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01) # Add weight_decay
    
    # Learning Rate Scheduling
    # Sometimes, as training progresses, the learning rate becomes too high, 
    # causing the model to "bounce around" a good solution and overfit.
    # It will automatically reduce the learning rate whenever the validation loss stops improving for a few epochs.
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)

    # --- Logic to Load Checkpoint and Resume Training ---
    start_epoch = 0
    best_valid_loss = float('inf')
    training_history = {
        'train_loss': [], 'train_accuracy': [], 'train_f1': [],
        'valid_loss': [], 'valid_em': [], 'valid_accuracy': [], 'valid_f1': []
    }

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming training from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # This is the new, dictionary-based format
            # Load model and optimizer states
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load training progress
            start_epoch = checkpoint['epoch'] + 1
            best_valid_loss = checkpoint['best_valid_loss']
        
        else:
            # This is the old format, which is just the model's state_dict
            print("Warning: Loading old format checkpoint. Contains model weights only.")
            print("Optimizer and epoch count will start from scratch.")
            model.load_state_dict(checkpoint)

        print(f"Resumed from Epoch {start_epoch}. Best validation loss so far: {best_valid_loss:.4f}")

    # Logic to load training history if provided ---
    if args.resume_history and os.path.exists(args.resume_history):
        print(f"Loading and resuming from history file: {args.resume_history}")
        with open(args.resume_history, 'r') as f:
            training_history = json.load(f)
        
        # --- NEW: Ensure all expected keys exist after loading ---
        # This makes the script compatible with older history files.
        training_history.setdefault('train_accuracy', [])
        training_history.setdefault('train_f1', [])
        training_history.setdefault('valid_accuracy', [])
        training_history.setdefault('valid_f1', [])
        
        # If history is loaded, it's the source of truth for epoch and best loss
        if training_history.get('valid_loss'): # Check if list is not empty
             best_valid_loss = min(training_history['valid_loss'])
        
        # The next epoch to start is the number of epochs already completed.
        start_epoch = len(training_history['train_loss'])        

        print(f"State from history file: Resuming from Epoch {start_epoch + 1}. Best validation loss: {best_valid_loss:.4f}")

    print("Model, Loss, and Optimizer initialized.")

    def decode_program(program_ids, single_graph_data):
        decoded_tokens = []
        for token_id in program_ids:
            token_id = token_id.item()
            if token_id == op_list.index('EOS'):
                break # Stop decoding after EOS
            if token_id == 0: # Skip padding
                continue

            if token_id < reserved_token_size:
                # It's a fixed vocabulary token
                decoded_tokens.append(full_vocab[token_id])
            else:
                # It's a pointer
                node_idx = token_id - reserved_token_size
                if node_idx < len(single_graph_data.tokens):
                    pointed_word = single_graph_data.tokens[node_idx]
                    pointed_loc = single_graph_data.node_locations[node_idx]
                    decoded_tokens.append(f"ptr_to(word='{pointed_word}', loc={pointed_loc})")
                else:
                    decoded_tokens.append("ptr_to(INVALID_NODE)")
        return " ".join(decoded_tokens)

    # --- 4. Training Loop ---
    print("\n--- Starting Training ---")
    for epoch in range(start_epoch, args.epochs):
        model.train() # Set the model to training mode
        total_train_loss = 0
        all_train_targets = []
        all_train_predictions = []
        
        # Iterate over the training data
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]"):
            batch = batch.to(device)
            # The target program is attached to the 'y' attribute by our custom Dataset
            target_program_flat = batch.y
            
            num_graphs_in_batch = batch.num_graphs
            target_program = target_program_flat.view(num_graphs_in_batch, -1)

            # --- Main Training Step ---
            optimizer.zero_grad()
            output_logits = model(batch, target_program)
            
            # Reshape for CrossEntropyLoss and calculate the loss
            loss = criterion(
                output_logits.view(-1, model.decoder.full_vocab_size),
                target_program.view(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping
            optimizer.step()
            
            total_train_loss += loss.item()

            # --- NEW: Collect predictions and targets for training metrics ---
            with torch.no_grad():
                # Get predictions from the logits used for loss calculation
                predicted_tokens = torch.argmax(output_logits, dim=-1)
                
                targets_flat = target_program.view(-1)
                preds_flat = predicted_tokens.view(-1)
                
                non_padding_mask = targets_flat != 0
                
                all_train_targets.extend(targets_flat[non_padding_mask].cpu().numpy())
                all_train_predictions.extend(preds_flat[non_padding_mask].cpu().numpy())

        # --- NEW: Calculate training metrics for the epoch ---
        avg_train_loss = total_train_loss / len(train_loader)
        train_token_accuracy = accuracy_score(all_train_targets, all_train_predictions)
        _, _, train_f1, _ = precision_recall_fscore_support(
            all_train_targets, all_train_predictions, average='macro', zero_division=0
        )
        training_history['train_loss'].append(avg_train_loss)
        training_history['train_accuracy'].append(train_token_accuracy)
        training_history['train_f1'].append(train_f1)

        # avg_train_loss = total_train_loss / len(train_loader)
        # training_history['train_loss'].append(avg_train_loss)

        # --- 5. Validation Loop ---
        model.eval() # Set the model to evaluation mode
        total_valid_loss = 0
        total_em = 0 # Exact Match counter
        all_valid_targets = []
        all_valid_predictions = []
        
        prediction_log_path = os.path.join(args.output_dir, f'{exp_id}_epoch_{epoch+1}_predictions.txt')
        with open(prediction_log_path, 'w', encoding='utf-8') as log_file:
            log_file.write("Table UID | Question UID | Target Program | Predicted Program | Target Token ID | Predicted Token ID\n")
            log_file.write("="*120 + "\n")

            with torch.no_grad():
                for batch in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]"):
                    batch = batch.to(device)
                    target_program_flat = batch.y

                    num_graphs_in_batch = batch.num_graphs
                    target_program = target_program_flat.view(num_graphs_in_batch, -1)

                    # Get the training loss (for comparison)
                    output_logits = model(batch, target_program)
                    loss = criterion(
                        output_logits.view(-1, model.decoder.full_vocab_size),
                        target_program.view(-1)
                    )
                    total_valid_loss += loss.item()
                    
                    # --- Calculate Exact Match (EM) Accuracy ---
                    predicted_program = model.predict(batch, max_length=args.program_length)
                    
                    # Compare predicted sequence with the target sequence
                    # We ignore padding when checking for equality
                    is_match = (predicted_program == target_program).all(dim=-1)
                    total_em += is_match.sum().item()

                    targets_flat = target_program.view(-1)
                    preds_flat = predicted_program.view(-1)
                    non_padding_mask = targets_flat != 0
                    all_valid_targets.extend(targets_flat[non_padding_mask].cpu().numpy())
                    all_valid_predictions.extend(preds_flat[non_padding_mask].cpu().numpy())

                    # Deconstruct the batch into a list of individual graphs
                    graph_list = batch.to_data_list()
                    for i in range(batch.num_graphs):
                        target_ids = target_program[i]
                        predicted_ids = predicted_program[i]
                        single_graph = graph_list[i]

                        # Decode both sequences into human-readable strings
                        decoded_target = decode_program(target_ids, single_graph)
                        decoded_predicted = decode_program(predicted_ids, single_graph)
                        
                        # Get the UIDs for the i-th graph from the batch object
                        tbl_uid = batch.table_uid[i]
                        qst_uid = batch.question_uid[i]

                        # Write the log line
                        log_file.write(f"{tbl_uid} | {qst_uid} | {decoded_target} | {decoded_predicted} | {str(target_ids.tolist())} | {str(predicted_ids.tolist())}\n")

        # avg_valid_loss = total_valid_loss / len(valid_loader)
        # valid_em_accuracy = total_em / len(valid_dataset)
        # training_history['valid_loss'].append(avg_valid_loss)
        # training_history['valid_em'].append(valid_em_accuracy)

        # print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | "
        #       f"Valid Loss: {avg_valid_loss:.4f} | Valid EM: {valid_em_accuracy:.4f}")
        # --- Calculate and store validation metrics ---
        avg_valid_loss = total_valid_loss / len(valid_loader)
        # automatically reduce learning rate when validation loss stops improving
        scheduler.step(avg_valid_loss)
        valid_em_accuracy = total_em / len(valid_dataset)
        valid_token_accuracy = accuracy_score(all_valid_targets, all_valid_predictions)
        _, _, valid_f1, _ = precision_recall_fscore_support(
            all_valid_targets, all_valid_predictions, average='macro', zero_division=0
        )
        training_history['valid_loss'].append(avg_valid_loss)
        training_history['valid_em'].append(valid_em_accuracy)
        training_history['valid_accuracy'].append(valid_token_accuracy)
        training_history['valid_f1'].append(valid_f1)

        # --- Updated Print Statement ---
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_token_accuracy:.4f}, F1: {train_f1:.4f} | "
              f"Valid Loss: {avg_valid_loss:.4f}, EM: {valid_em_accuracy:.4f}, Acc: {valid_token_accuracy:.4f}, F1: {valid_f1:.4f}")
        print(f"Predictions for this epoch saved to: {prediction_log_path}")

        # --- 6. Model Checkpointing ---
        if avg_valid_loss < best_valid_loss:
            print(f"Validation loss improved ({best_valid_loss:.4f} --> {avg_valid_loss:.4f}). Saving model...")
            best_valid_loss = avg_valid_loss
            # torch.save(model.state_dict(), os.path.join(args.output_dir, f'{exp_id}_best_model_{args.text_encoder}.pt'))
            save_path = os.path.join(args.output_dir, f'{exp_id}_best_model_{args.text_encoder}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_valid_loss': best_valid_loss,
            }, save_path)

        with open(os.path.join(args.output_dir, f'{exp_id}_training_history_{args.text_encoder}.json'), 'w') as f:
            json.dump(training_history, f)

    print("--- Training Finished ---")
    # Save the training history for later visualization
    with open(os.path.join(args.output_dir, f'{exp_id}_training_history_{args.text_encoder}.json'), 'w') as f:
        json.dump(training_history, f)

def train_blaze(args):
    """
    The main function to handle the model training process. This mode of training only check EM every 20 epoch.
    """
    # --- 1. Setup ---
    print("--- Starting Training Setup ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cuda')
    print(f"Using device: {device}")

    # experiment id
    exp_id = timestamp_exp_start+"_"+args.exp_id if timestamp_exp_start!=args.exp_id else args.exp_id

    # Define the program vocabulary.
    op_list = ['GO', 'EOS', '(', ')', ',', 
               "retrieve", "addition", "subtraction", "multiplication", "division", 
                "exponential", "greater", "smaller", "equal", "sum", "average", 
                "minimum", "maximum", "maximum_n", "minimum_n", "count", "count_if_equal", 
                "count_if_less", "count_if_greater", "count_if_geq", "count_if_leq", 
                "count_if_notequal", "sum_if_equal", "sum_if_less", "sum_if_greater", "sum_if_geq", 
                "sum_if_leq", "sum_if_notequal", "filter_if_equal", "filter_if_less",
                "filter_if_greater", "filter_if_geq", "filter_if_leq", "filter_if_notequal",
                "trace_column", "trace_row", "unique"] + [f'#{i}' for i in range(1000)]
    const_list = [f'const_{i}' for i in range(1000)]
    full_vocab = op_list + const_list
    reserved_token_size = len(full_vocab)

    # --- 2. Data Loading ---
    
    # Create PyTorch Geometric Dataset objects.
    train_dataset = ProgramDataset(
        root=os.path.join(args.output_dir, 'train'),
        json_files=args.train_file,
        op_list=op_list, const_list=const_list, program_length=args.program_length,
        tokenizer=args.text_encoder
    )
    valid_dataset = ProgramDataset(
        root=os.path.join(args.output_dir, 'valid'),
        json_files=args.valid_file,
        op_list=op_list, const_list=const_list, program_length=args.program_length,
        tokenizer=args.text_encoder
    )
    # test_dataset = ProgramDataset(
    #     root=os.path.join(args.output_dir, 'test'),
    #     json_files=args.test_file,
    #     op_list=op_list, const_list=const_list, program_length=args.program_length,
    #     tokenizer=args.text_encoder
    # )

    vocab_size = train_dataset.tokenizer.vocab_size

    # Create DataLoaders to handle batching and shuffling.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=3)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,num_workers=3)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    print("DataLoaders created successfully.")

    if args.text_encoder == 'deepseek_qwen':
        # language_model = AutoModel.from_pretrained('FacebookAI/xlm-roberta-base')
        local_model_path = "deepseek.qwen"
        language_model = AutoModel.from_pretrained(local_model_path)
    elif args.text_encoder == 'gemma3':
        # language_model = AutoModel.from_pretrained('FacebookAI/xlm-roberta-base')
        local_model_path = "gemma3.model"
        language_model = AutoModel.from_pretrained(local_model_path)
    # elif args.text_encoder == 'gemma3.finetune':
    #     # language_model = AutoModel.from_pretrained('FacebookAI/xlm-roberta-base')
    #     local_model_path = "gemma3.270m.finetune.embedding.indompf"
    #     # language_model = AutoModel.from_pretrained(local_model_path)
    #     embedding_dimension = 640
    elif args.text_encoder == 'gemma3.finetune':
        # language_model = AutoModel.from_pretrained('FacebookAI/xlm-roberta-base')
        # local_model_path = "gemma3.270m.finetune.embedding.indompf"
        local_model_path = "gemma3.embedding.finetune.mnr"
        # language_model = AutoModel.from_pretrained(local_model_path)
        # embedding_dimension = language_model.config.hidden_size
        embedding_dimension=768
        # del language_model
    elif args.text_encoder == 'xlm_roberta_base':
        # language_model = AutoModel.from_pretrained('FacebookAI/xlm-roberta-base')
        local_model_path = "xlm.roberta.base"
        language_model = AutoModel.from_pretrained(local_model_path)
    elif args.text_encoder == 'xlm_roberta_large':
        local_model_path = "xlm.roberta.large"
        language_model = AutoModel.from_pretrained(local_model_path)
    elif args.text_encoder == 'indobert_base_uncased':
        local_model_path = "indobert.base.uncased"
        language_model = AutoModel.from_pretrained(local_model_path)
    elif args.text_encoder == 'indobert_large_p2':
        language_model = AutoModel.from_pretrained('indobenchmark/indobert-large-p2')

    # Access the configuration and retrieve the hidden_size
    

    # --- 3. Model, Loss, and Optimizer ---
    # Instantiate the full encoder-decoder model.
    encoder = GraphEncoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dimension,
        text_encoder_name=args.text_encoder,
        hidden_channels=args.HIDDEN_DIM,
        num_layers=args.encoder_layer,
        num_heads=args.encoder_head,
        num_relations=NUM_RELATIONS
    )
    decoder = TransformerProgramDecoder(
        hidden_size=args.HIDDEN_DIM, program_length=args.program_length, op_list=op_list, const_list=const_list,
        max_input_length=MAX_NODES, # Use the max nodes from the padded batch
        num_layers = args.decoder_layer,
        num_heads = args.decoder_head,
    )
    model = IndoMPFGATe(encoder, decoder).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    print("\nFull Transformer-based GraphProgramGenerator model instantiated successfully.")    
    # model = GraphProgramGenerator(None, None) # Placeholder
    
    # The loss function: CrossEntropyLoss is ideal for this token classification task.
    # We tell it to ignore the padding token (ID 0) when calculating the loss.
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # The optimizer: AdamW is a robust choice for Transformer-based models.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01) # Add weight_decay
    
    # Learning Rate Scheduling
    # Sometimes, as training progresses, the learning rate becomes too high, 
    # causing the model to "bounce around" a good solution and overfit.
    # It will automatically reduce the learning rate whenever the validation loss stops improving for a few epochs.
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)

    # --- Logic to Load Checkpoint and Resume Training ---
    start_epoch = 0
    best_valid_loss = float('inf')
    training_history = {
        'train_loss': [], 'train_accuracy': [], 'train_f1': [],
        'valid_loss': [], 'valid_em': [], 'valid_accuracy': [], 'valid_f1': []
    }

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming training from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # This is the new, dictionary-based format
            # Load model and optimizer states
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load training progress
            start_epoch = checkpoint['epoch'] + 1
            best_valid_loss = checkpoint['best_valid_loss']
        
        else:
            # This is the old format, which is just the model's state_dict
            print("Warning: Loading old format checkpoint. Contains model weights only.")
            print("Optimizer and epoch count will start from scratch.")
            model.load_state_dict(checkpoint)

        print(f"Resumed from Epoch {start_epoch}. Best validation loss so far: {best_valid_loss:.4f}")

    # Logic to load training history if provided ---
    if args.resume_history and os.path.exists(args.resume_history):
        print(f"Loading and resuming from history file: {args.resume_history}")
        with open(args.resume_history, 'r') as f:
            training_history = json.load(f)
        
        # --- NEW: Ensure all expected keys exist after loading ---
        # This makes the script compatible with older history files.
        training_history.setdefault('train_accuracy', [])
        training_history.setdefault('train_f1', [])
        training_history.setdefault('valid_accuracy', [])
        training_history.setdefault('valid_f1', [])
        
        # If history is loaded, it's the source of truth for epoch and best loss
        if training_history.get('valid_loss'): # Check if list is not empty
             best_valid_loss = min(training_history['valid_loss'])
        
        # The next epoch to start is the number of epochs already completed.
        start_epoch = len(training_history['train_loss'])        

        print(f"State from history file: Resuming from Epoch {start_epoch + 1}. Best validation loss: {best_valid_loss:.4f}")

    print("Model, Loss, and Optimizer initialized.")

    def decode_program(program_ids, single_graph_data):
        decoded_tokens = []
        for token_id in program_ids:
            token_id = token_id.item()
            if token_id == op_list.index('EOS'):
                break # Stop decoding after EOS
            if token_id == 0: # Skip padding
                continue

            if token_id < reserved_token_size:
                # It's a fixed vocabulary token
                decoded_tokens.append(full_vocab[token_id])
            else:
                # It's a pointer
                node_idx = token_id - reserved_token_size
                if node_idx < len(single_graph_data.tokens):
                    pointed_word = single_graph_data.tokens[node_idx]
                    pointed_loc = single_graph_data.node_locations[node_idx]
                    decoded_tokens.append(f"ptr_to(word='{pointed_word}', loc={pointed_loc})")
                else:
                    decoded_tokens.append("ptr_to(INVALID_NODE)")
        return " ".join(decoded_tokens)

    # --- 4. Training Loop ---
    print("\n--- Starting Training ---")
    for epoch in range(start_epoch, args.epochs):
        model.train() # Set the model to training mode
        total_train_loss = 0
        all_train_targets = []
        all_train_predictions = []
        
        # Iterate over the training data
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]"):
            batch = batch.to(device)
            # The target program is attached to the 'y' attribute by our custom Dataset
            target_program_flat = batch.y
            
            num_graphs_in_batch = batch.num_graphs
            target_program = target_program_flat.view(num_graphs_in_batch, -1)

            # --- Main Training Step ---
            optimizer.zero_grad()
            output_logits = model(batch, target_program)
            
            # Reshape for CrossEntropyLoss and calculate the loss
            loss = criterion(
                output_logits.view(-1, model.decoder.full_vocab_size),
                target_program.view(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping
            optimizer.step()
            
            total_train_loss += loss.item()

            # --- NEW: Collect predictions and targets for training metrics ---
            with torch.no_grad():
                # Get predictions from the logits used for loss calculation
                predicted_tokens = torch.argmax(output_logits, dim=-1)
                
                targets_flat = target_program.view(-1)
                preds_flat = predicted_tokens.view(-1)
                
                non_padding_mask = targets_flat != 0
                
                all_train_targets.extend(targets_flat[non_padding_mask].cpu().numpy())
                all_train_predictions.extend(preds_flat[non_padding_mask].cpu().numpy())

        # --- NEW: Calculate training metrics for the epoch ---
        avg_train_loss = total_train_loss / len(train_loader)
        train_token_accuracy = accuracy_score(all_train_targets, all_train_predictions)
        _, _, train_f1, _ = precision_recall_fscore_support(
            all_train_targets, all_train_predictions, average='macro', zero_division=0
        )
        training_history['train_loss'].append(avg_train_loss)
        training_history['train_accuracy'].append(train_token_accuracy)
        training_history['train_f1'].append(train_f1)

        # avg_train_loss = total_train_loss / len(train_loader)
        # training_history['train_loss'].append(avg_train_loss)

        # --- 5. Validation Loop ---
        model.eval() # Set the model to evaluation mode
        total_valid_loss = 0
        total_em = 0 # Exact Match counter
        all_valid_targets = []
        all_valid_predictions = []

        # only calculate epoch and see prediction in multiple 20 epoch 
        if epoch+1 % 20==0:
        
            prediction_log_path = os.path.join(args.output_dir, f'{exp_id}_epoch_{epoch+1}_predictions.txt')
            with open(prediction_log_path, 'w', encoding='utf-8') as log_file:
                log_file.write("Table UID | Question UID | Target Program | Predicted Program | Target Token ID | Predicted Token ID\n")
                log_file.write("="*120 + "\n")

                with torch.no_grad():
                    for batch in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]"):
                        batch = batch.to(device)
                        target_program_flat = batch.y

                        num_graphs_in_batch = batch.num_graphs
                        target_program = target_program_flat.view(num_graphs_in_batch, -1)

                        # Get the training loss (for comparison)
                        output_logits = model(batch, target_program)
                        loss = criterion(
                            output_logits.view(-1, model.decoder.full_vocab_size),
                            target_program.view(-1)
                        )
                        total_valid_loss += loss.item()
                        
                        # --- Calculate Exact Match (EM) Accuracy ---
                        predicted_program = model.predict(batch, max_length=args.program_length)
                        
                        # Compare predicted sequence with the target sequence
                        # We ignore padding when checking for equality
                        is_match = (predicted_program == target_program).all(dim=-1)
                        total_em += is_match.sum().item()

                        targets_flat = target_program.view(-1)
                        preds_flat = predicted_program.view(-1)
                        non_padding_mask = targets_flat != 0
                        all_valid_targets.extend(targets_flat[non_padding_mask].cpu().numpy())
                        all_valid_predictions.extend(preds_flat[non_padding_mask].cpu().numpy())

                        # Deconstruct the batch into a list of individual graphs
                        graph_list = batch.to_data_list()
                        for i in range(batch.num_graphs):
                            target_ids = target_program[i]
                            predicted_ids = predicted_program[i]
                            single_graph = graph_list[i]

                            # Decode both sequences into human-readable strings
                            decoded_target = decode_program(target_ids, single_graph)
                            decoded_predicted = decode_program(predicted_ids, single_graph)
                            
                            # Get the UIDs for the i-th graph from the batch object
                            tbl_uid = batch.table_uid[i]
                            qst_uid = batch.question_uid[i]

                            # Write the log line
                            log_file.write(f"{tbl_uid} | {qst_uid} | {decoded_target} | {decoded_predicted} | {str(target_ids.tolist())} | {str(predicted_ids.tolist())}\n")

            print(f"Predictions for this epoch saved to: {prediction_log_path}")

        else:
            with torch.no_grad():
                for batch in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]"):
                    batch = batch.to(device)
                    target_program_flat = batch.y

                    num_graphs_in_batch = batch.num_graphs
                    target_program = target_program_flat.view(num_graphs_in_batch, -1)

                    # Get the training loss (for comparison)
                    output_logits = model(batch, target_program)
                    loss = criterion(
                        output_logits.view(-1, model.decoder.full_vocab_size),
                        target_program.view(-1)
                    )
                    total_valid_loss += loss.item()

                    predicted_program = torch.argmax(output_logits, dim=-1)                    

                    targets_flat = target_program.view(-1)
                    preds_flat = predicted_program.view(-1)
                    non_padding_mask = targets_flat != 0
                    all_valid_targets.extend(targets_flat[non_padding_mask].cpu().numpy())
                    all_valid_predictions.extend(preds_flat[non_padding_mask].cpu().numpy())

        # avg_valid_loss = total_valid_loss / len(valid_loader)
        # valid_em_accuracy = total_em / len(valid_dataset)
        # training_history['valid_loss'].append(avg_valid_loss)
        # training_history['valid_em'].append(valid_em_accuracy)

        # print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | "
        #       f"Valid Loss: {avg_valid_loss:.4f} | Valid EM: {valid_em_accuracy:.4f}")
        # --- Calculate and store validation metrics ---
        avg_valid_loss = total_valid_loss / len(valid_loader)
        # automatically reduce learning rate when validation loss stops improving
        scheduler.step(avg_valid_loss)
        valid_em_accuracy = total_em / len(valid_dataset)
        valid_token_accuracy = accuracy_score(all_valid_targets, all_valid_predictions)
        _, _, valid_f1, _ = precision_recall_fscore_support(
            all_valid_targets, all_valid_predictions, average='macro', zero_division=0
        )
        training_history['valid_loss'].append(avg_valid_loss)
        training_history['valid_em'].append(valid_em_accuracy)
        training_history['valid_accuracy'].append(valid_token_accuracy)
        training_history['valid_f1'].append(valid_f1)

        # --- Updated Print Statement ---
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_token_accuracy:.4f}, F1: {train_f1:.4f} | "
              f"Valid Loss: {avg_valid_loss:.4f}, EM: {valid_em_accuracy:.4f}, Acc: {valid_token_accuracy:.4f}, F1: {valid_f1:.4f}")

        # --- 6. Model Checkpointing ---
        if avg_valid_loss < best_valid_loss:
            print(f"Validation loss improved ({best_valid_loss:.4f} --> {avg_valid_loss:.4f}). Saving model...")
            best_valid_loss = avg_valid_loss
            # torch.save(model.state_dict(), os.path.join(args.output_dir, f'{exp_id}_best_model_{args.text_encoder}.pt'))
            save_path = os.path.join(args.output_dir, f'{exp_id}_best_model_{args.text_encoder}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_valid_loss': best_valid_loss,
            }, save_path)
        
        save_path_2 = os.path.join(args.output_dir, f'{exp_id}_checkpoint_model_epoch_{epoch+1}_{args.text_encoder}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_valid_loss': best_valid_loss,
        }, save_path_2)

        with open(os.path.join(args.output_dir, f'{exp_id}_training_history_{args.text_encoder}.json'), 'w') as f:
            json.dump(training_history, f)

    print("--- Training Finished ---")
    # Save the training history for later visualization
    with open(os.path.join(args.output_dir, f'{exp_id}_training_history_{args.text_encoder}.json'), 'w') as f:
        json.dump(training_history, f)

def train_blaze_auto(args):
    """
    The main function to handle the model training process. This mode of training only check EM every 20 epoch.
    """
    # --- 1. Setup ---
    print("--- Starting Training Setup ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cuda')
    print(f"Using device: {device}")

    # experiment id
    exp_id = timestamp_exp_start+"_"+args.exp_id if timestamp_exp_start!=args.exp_id else args.exp_id

    # Define the program vocabulary.
    op_list = ['GO', 'EOS', '(', ')', ',', 
               "retrieve", "addition", "subtraction", "multiplication", "division", 
                "exponential", "greater", "smaller", "equal", "sum", "average", 
                "minimum", "maximum", "maximum_n", "minimum_n", "count", "count_if_equal", 
                "count_if_less", "count_if_greater", "count_if_geq", "count_if_leq", 
                "count_if_notequal", "sum_if_equal", "sum_if_less", "sum_if_greater", "sum_if_geq", 
                "sum_if_leq", "sum_if_notequal", "filter_if_equal", "filter_if_less",
                "filter_if_greater", "filter_if_geq", "filter_if_leq", "filter_if_notequal",
                "trace_column", "trace_row", "unique"] + [f'#{i}' for i in range(1000)]
    const_list = [f'const_{i}' for i in range(1000)]
    full_vocab = op_list + const_list
    reserved_token_size = len(full_vocab)

    # --- 2. Data Loading ---
    
    # Create PyTorch Geometric Dataset objects.
    train_dataset = ProgramDataset(
        root=os.path.join(args.output_dir, 'train'),
        json_files=args.train_file,
        op_list=op_list, const_list=const_list, program_length=args.program_length,
        tokenizer=args.text_encoder
    )
    valid_dataset = ProgramDataset(
        root=os.path.join(args.output_dir, 'valid'),
        json_files=args.valid_file,
        op_list=op_list, const_list=const_list, program_length=args.program_length,
        tokenizer=args.text_encoder
    )
    test_dataset = ProgramDataset(
        root=os.path.join(args.output_dir, 'test'),
        json_files=args.test_file,
        op_list=op_list, const_list=const_list, program_length=args.program_length,
        tokenizer=args.text_encoder
    )

    vocab_size = train_dataset.tokenizer.vocab_size

    # Create DataLoaders to handle batching and shuffling.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=3)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,num_workers=3)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    print("DataLoaders created successfully.")

    if args.text_encoder == 'deepseek_qwen':
        # language_model = AutoModel.from_pretrained('FacebookAI/xlm-roberta-base')
        local_model_path = "deepseek.qwen"
        language_model = AutoModel.from_pretrained(local_model_path)
    elif args.text_encoder == 'gemma3':
        # language_model = AutoModel.from_pretrained('FacebookAI/xlm-roberta-base')
        local_model_path = "gemma3.model"
        language_model = AutoModel.from_pretrained(local_model_path)
    # elif args.text_encoder == 'gemma3.finetune':
    #     # language_model = AutoModel.from_pretrained('FacebookAI/xlm-roberta-base')
    #     local_model_path = "gemma3.270m.finetune.embedding.indompf"
    #     # language_model = AutoModel.from_pretrained(local_model_path)
    #     embedding_dimension = 640
    elif args.text_encoder == 'gemma3.finetune':
        # language_model = AutoModel.from_pretrained('FacebookAI/xlm-roberta-base')
        # local_model_path = "gemma3.270m.finetune.embedding.indompf"
        local_model_path = "gemma3.embedding.finetune.mnr"
        language_model = AutoModel.from_pretrained(local_model_path)
        embedding_dimension = language_model.config.hidden_size
        del language_model
    elif args.text_encoder == 'xlm_roberta_base':
        # language_model = AutoModel.from_pretrained('FacebookAI/xlm-roberta-base')
        local_model_path = "xlm.roberta.base"
        language_model = AutoModel.from_pretrained(local_model_path)
    elif args.text_encoder == 'xlm_roberta_large':
        local_model_path = "xlm.roberta.large"
        language_model = AutoModel.from_pretrained(local_model_path)
    elif args.text_encoder == 'indobert_base_uncased':
        local_model_path = "indobert.base.uncased"
        language_model = AutoModel.from_pretrained(local_model_path)
    elif args.text_encoder == 'indobert_large_p2':
        language_model = AutoModel.from_pretrained('indobenchmark/indobert-large-p2')

    # Access the configuration and retrieve the hidden_size
    

    # --- 3. Model, Loss, and Optimizer ---
    # Instantiate the full encoder-decoder model.
    encoder = GraphEncoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dimension,
        text_encoder_name=args.text_encoder,
        hidden_channels=args.HIDDEN_DIM,
        num_layers=args.encoder_layer,
        num_heads=args.encoder_head,
        num_relations=NUM_RELATIONS
    )
    decoder = TransformerProgramDecoder(
        hidden_size=args.HIDDEN_DIM, program_length=args.program_length, op_list=op_list, const_list=const_list,
        max_input_length=MAX_NODES, # Use the max nodes from the padded batch
        num_layers = args.decoder_layer,
        num_heads = args.decoder_head,
    )
    model = IndoMPFGATe(encoder, decoder).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    print("\nFull Transformer-based GraphProgramGenerator model instantiated successfully.")    
    # model = GraphProgramGenerator(None, None) # Placeholder
    
    # The loss function: CrossEntropyLoss is ideal for this token classification task.
    # We tell it to ignore the padding token (ID 0) when calculating the loss.
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # The optimizer: AdamW is a robust choice for Transformer-based models.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01) # Add weight_decay
    
    # Learning Rate Scheduling
    # Sometimes, as training progresses, the learning rate becomes too high, 
    # causing the model to "bounce around" a good solution and overfit.
    # It will automatically reduce the learning rate whenever the validation loss stops improving for a few epochs.
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)

    # --- Logic to Load Checkpoint and Resume Training ---
    start_epoch = 0
    best_valid_loss = float('inf')
    training_history = {
        'train_loss': [], 'train_accuracy': [], 'train_f1': [],
        'valid_loss': [], 'valid_em': [], 'valid_accuracy': [], 'valid_f1': []
    }
    
    history_search_pattern = os.path.join(args.output_dir, '*.json')
    resume_search_pattern = os.path.join(args.output_dir, '*.pt')

    # Get a list of all JSON files in the directory
    list_of_history_files = glob.glob(history_search_pattern)
    list_of_resume_files = glob.glob(resume_search_pattern)
    
    # Find the most recently modified file using os.path.getmtime
    # and the max() function with a key for sorting by modification time
    resume_history = max(list_of_history_files, key=os.path.getmtime)
    resume_from = max(list_of_resume_files, key=os.path.getmtime)

    if resume_from and os.path.exists(resume_from):
        print(f"Resuming training from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # This is the new, dictionary-based format
            # Load model and optimizer states
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load training progress
            start_epoch = checkpoint['epoch'] + 1
            best_valid_loss = checkpoint['best_valid_loss']
        
        else:
            # This is the old format, which is just the model's state_dict
            print("Warning: Loading old format checkpoint. Contains model weights only.")
            print("Optimizer and epoch count will start from scratch.")
            model.load_state_dict(checkpoint)

        print(f"Resumed from Epoch {start_epoch}. Best validation loss so far: {best_valid_loss:.4f}")

    # Logic to load training history if provided ---
    if resume_history and os.path.exists(resume_history):
        print(f"Loading and resuming from history file: {resume_history}")
        with open(resume_history, 'r') as f:
            training_history = json.load(f)
        
        # --- NEW: Ensure all expected keys exist after loading ---
        # This makes the script compatible with older history files.
        training_history.setdefault('train_accuracy', [])
        training_history.setdefault('train_f1', [])
        training_history.setdefault('valid_accuracy', [])
        training_history.setdefault('valid_f1', [])
        
        # If history is loaded, it's the source of truth for epoch and best loss
        if training_history.get('valid_loss'): # Check if list is not empty
             best_valid_loss = min(training_history['valid_loss'])
        
        # The next epoch to start is the number of epochs already completed.
        start_epoch = len(training_history['train_loss'])        

        print(f"State from history file: Resuming from Epoch {start_epoch + 1}. Best validation loss: {best_valid_loss:.4f}")

    print("Model, Loss, and Optimizer initialized.")

    def decode_program(program_ids, single_graph_data):
        decoded_tokens = []
        for token_id in program_ids:
            token_id = token_id.item()
            if token_id == op_list.index('EOS'):
                break # Stop decoding after EOS
            if token_id == 0: # Skip padding
                continue

            if token_id < reserved_token_size:
                # It's a fixed vocabulary token
                decoded_tokens.append(full_vocab[token_id])
            else:
                # It's a pointer
                node_idx = token_id - reserved_token_size
                if node_idx < len(single_graph_data.tokens):
                    pointed_word = single_graph_data.tokens[node_idx]
                    pointed_loc = single_graph_data.node_locations[node_idx]
                    decoded_tokens.append(f"ptr_to(word='{pointed_word}', loc={pointed_loc})")
                else:
                    decoded_tokens.append("ptr_to(INVALID_NODE)")
        return " ".join(decoded_tokens)

    # --- 4. Training Loop ---
    print("\n--- Starting Training ---")
    for epoch in range(start_epoch, args.epochs):
        model.train() # Set the model to training mode
        total_train_loss = 0
        all_train_targets = []
        all_train_predictions = []
        
        # Iterate over the training data
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]"):
            batch = batch.to(device)
            # The target program is attached to the 'y' attribute by our custom Dataset
            target_program_flat = batch.y
            
            num_graphs_in_batch = batch.num_graphs
            target_program = target_program_flat.view(num_graphs_in_batch, -1)

            # --- Main Training Step ---
            optimizer.zero_grad()
            output_logits = model(batch, target_program)
            
            # Reshape for CrossEntropyLoss and calculate the loss
            loss = criterion(
                output_logits.view(-1, model.decoder.full_vocab_size),
                target_program.view(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping
            optimizer.step()
            
            total_train_loss += loss.item()

            # --- NEW: Collect predictions and targets for training metrics ---
            with torch.no_grad():
                # Get predictions from the logits used for loss calculation
                predicted_tokens = torch.argmax(output_logits, dim=-1)
                
                targets_flat = target_program.view(-1)
                preds_flat = predicted_tokens.view(-1)
                
                non_padding_mask = targets_flat != 0
                
                all_train_targets.extend(targets_flat[non_padding_mask].cpu().numpy())
                all_train_predictions.extend(preds_flat[non_padding_mask].cpu().numpy())

        # --- NEW: Calculate training metrics for the epoch ---
        avg_train_loss = total_train_loss / len(train_loader)
        train_token_accuracy = accuracy_score(all_train_targets, all_train_predictions)
        _, _, train_f1, _ = precision_recall_fscore_support(
            all_train_targets, all_train_predictions, average='macro', zero_division=0
        )
        training_history['train_loss'].append(avg_train_loss)
        training_history['train_accuracy'].append(train_token_accuracy)
        training_history['train_f1'].append(train_f1)

        # avg_train_loss = total_train_loss / len(train_loader)
        # training_history['train_loss'].append(avg_train_loss)

        # --- 5. Validation Loop ---
        model.eval() # Set the model to evaluation mode
        total_valid_loss = 0
        total_em = 0 # Exact Match counter
        all_valid_targets = []
        all_valid_predictions = []

        # only calculate epoch and see prediction in multiple 50 epoch 
        if epoch+1 % 70==0:
        
            prediction_log_path = os.path.join(args.output_dir, f'{exp_id}_epoch_{epoch+1}_predictions.txt')
            with open(prediction_log_path, 'w', encoding='utf-8') as log_file:
                log_file.write("Table UID | Question UID | Target Program | Predicted Program | Target Token ID | Predicted Token ID\n")
                log_file.write("="*120 + "\n")

                with torch.no_grad():
                    for batch in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]"):
                        batch = batch.to(device)
                        target_program_flat = batch.y

                        num_graphs_in_batch = batch.num_graphs
                        target_program = target_program_flat.view(num_graphs_in_batch, -1)

                        # Get the training loss (for comparison)
                        output_logits = model(batch, target_program)
                        loss = criterion(
                            output_logits.view(-1, model.decoder.full_vocab_size),
                            target_program.view(-1)
                        )
                        total_valid_loss += loss.item()
                        
                        # --- Calculate Exact Match (EM) Accuracy ---
                        predicted_program = model.predict(batch, max_length=args.program_length)
                        
                        # Compare predicted sequence with the target sequence
                        # We ignore padding when checking for equality
                        is_match = (predicted_program == target_program).all(dim=-1)
                        total_em += is_match.sum().item()

                        targets_flat = target_program.view(-1)
                        preds_flat = predicted_program.view(-1)
                        non_padding_mask = targets_flat != 0
                        all_valid_targets.extend(targets_flat[non_padding_mask].cpu().numpy())
                        all_valid_predictions.extend(preds_flat[non_padding_mask].cpu().numpy())

                        # Deconstruct the batch into a list of individual graphs
                        graph_list = batch.to_data_list()
                        for i in range(batch.num_graphs):
                            target_ids = target_program[i]
                            predicted_ids = predicted_program[i]
                            single_graph = graph_list[i]

                            # Decode both sequences into human-readable strings
                            decoded_target = decode_program(target_ids, single_graph)
                            decoded_predicted = decode_program(predicted_ids, single_graph)
                            
                            # Get the UIDs for the i-th graph from the batch object
                            tbl_uid = batch.table_uid[i]
                            qst_uid = batch.question_uid[i]

                            # Write the log line
                            log_file.write(f"{tbl_uid} | {qst_uid} | {decoded_target} | {decoded_predicted} | {str(target_ids.tolist())} | {str(predicted_ids.tolist())}\n")

            print(f"Predictions for this epoch saved to: {prediction_log_path}")

        else:
            with torch.no_grad():
                for batch in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]"):
                    batch = batch.to(device)
                    target_program_flat = batch.y

                    num_graphs_in_batch = batch.num_graphs
                    target_program = target_program_flat.view(num_graphs_in_batch, -1)

                    # Get the training loss (for comparison)
                    output_logits = model(batch, target_program)
                    loss = criterion(
                        output_logits.view(-1, model.decoder.full_vocab_size),
                        target_program.view(-1)
                    )
                    total_valid_loss += loss.item()

                    predicted_program = torch.argmax(output_logits, dim=-1)                    

                    targets_flat = target_program.view(-1)
                    preds_flat = predicted_program.view(-1)
                    non_padding_mask = targets_flat != 0
                    all_valid_targets.extend(targets_flat[non_padding_mask].cpu().numpy())
                    all_valid_predictions.extend(preds_flat[non_padding_mask].cpu().numpy())

        # avg_valid_loss = total_valid_loss / len(valid_loader)
        # valid_em_accuracy = total_em / len(valid_dataset)
        # training_history['valid_loss'].append(avg_valid_loss)
        # training_history['valid_em'].append(valid_em_accuracy)

        # print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | "
        #       f"Valid Loss: {avg_valid_loss:.4f} | Valid EM: {valid_em_accuracy:.4f}")
        # --- Calculate and store validation metrics ---
        avg_valid_loss = total_valid_loss / len(valid_loader)
        # automatically reduce learning rate when validation loss stops improving
        scheduler.step(avg_valid_loss)
        valid_em_accuracy = total_em / len(valid_dataset)
        valid_token_accuracy = accuracy_score(all_valid_targets, all_valid_predictions)
        _, _, valid_f1, _ = precision_recall_fscore_support(
            all_valid_targets, all_valid_predictions, average='macro', zero_division=0
        )
        training_history['valid_loss'].append(avg_valid_loss)
        training_history['valid_em'].append(valid_em_accuracy)
        training_history['valid_accuracy'].append(valid_token_accuracy)
        training_history['valid_f1'].append(valid_f1)

        # --- Updated Print Statement ---
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_token_accuracy:.4f}, F1: {train_f1:.4f} | "
              f"Valid Loss: {avg_valid_loss:.4f}, EM: {valid_em_accuracy:.4f}, Acc: {valid_token_accuracy:.4f}, F1: {valid_f1:.4f}")

        # --- 6. Model Checkpointing ---
        if avg_valid_loss < best_valid_loss:
            print(f"Validation loss improved ({best_valid_loss:.4f} --> {avg_valid_loss:.4f}). Saving model...")
            best_valid_loss = avg_valid_loss
            # torch.save(model.state_dict(), os.path.join(args.output_dir, f'{exp_id}_best_model_{args.text_encoder}.pt'))
            save_path = os.path.join(args.output_dir, f'{exp_id}_best_model_{args.text_encoder}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_valid_loss': best_valid_loss,
            }, save_path)
        
        save_path_2 = os.path.join(args.output_dir, f'{exp_id}_checkpoint_model_epoch_{epoch+1}_{args.text_encoder}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_valid_loss': best_valid_loss,
        }, save_path_2)

        with open(os.path.join(args.output_dir, f'{exp_id}_training_history_{args.text_encoder}.json'), 'w') as f:
            json.dump(training_history, f)

    print("--- Training Finished ---")
    # Save the training history for later visualization
    with open(os.path.join(args.output_dir, f'{exp_id}_training_history_{args.text_encoder}.json'), 'w') as f:
        json.dump(training_history, f)

def load_model_for_inference(args, device):
    """
    Helper function to load the model and its checkpoint for testing or prediction.
    """
    # Define the program vocabulary (must be identical to training)
    op_list = ['GO', 'EOS', '(', ')', ',', "retrieve", "addition", "subtraction", "multiplication", "division", "exponential", "greater", "smaller", "equal", "sum", "average", "minimum", "maximum", "maximum_n", "minimum_n", "count", "count_if_equal", "count_if_less", "count_if_greater", "count_if_geq", "count_if_leq", "count_if_notequal", "sum_if_equal", "sum_if_less", "sum_if_greater", "sum_if_geq", "sum_if_leq", "sum_if_notequal", "filter_if_equal", "filter_if_less", "filter_if_greater", "filter_if_geq", "filter_if_leq", "filter_if_notequal", "trace_column", "trace_row", "unique"] + [f'#{i}' for i in range(1000)]
    const_list = [f'const_{i}' for i in range(1000)]

    # Load the language model to get its configuration
    if args.text_encoder == 'deepseek_qwen':
        # language_model = AutoModel.from_pretrained('FacebookAI/xlm-roberta-base')
        local_model_path = "deepseek.qwen"
        language_model = AutoModel.from_pretrained(local_model_path)
    elif args.text_encoder == 'gemma3':
        # language_model = AutoModel.from_pretrained('FacebookAI/xlm-roberta-base')
        local_model_path = "gemma3.model"
        language_model = AutoModel.from_pretrained(local_model_path)
    elif args.text_encoder == 'gemma3.finetune':
        # language_model = AutoModel.from_pretrained('FacebookAI/xlm-roberta-base')
        # local_model_path = "gemma3.270m.finetune.embedding.indompf"
        local_model_path = "gemma3.embedding.finetune.mnr"
        language_model = AutoModel.from_pretrained(local_model_path)
        embedding_dimension = language_model.config.hidden_size
        # del language_model
        # embedding_dimension=768
    elif args.text_encoder == 'xlm_roberta_base':
        # language_model = AutoModel.from_pretrained('FacebookAI/xlm-roberta-base')
        local_model_path = "xlm.roberta.base"
        language_model = AutoModel.from_pretrained(local_model_path)
    elif args.text_encoder == 'xlm_roberta_large':
        local_model_path = "xlm.roberta.large"
        language_model = AutoModel.from_pretrained(local_model_path)
    elif args.text_encoder == 'indobert_base_uncased':
        local_model_path = "indobert.base.uncased"
        language_model = AutoModel.from_pretrained(local_model_path)
    elif args.text_encoder == 'indobert_large_p2':
        language_model = AutoModel.from_pretrained('indobenchmark/indobert-large-p2')
    # embedding_dimension = language_model.config.hidden_size
    vocab_size = language_model.config.vocab_size
    del language_model # Free up memory

    # Instantiate the model architecture
    encoder = GraphEncoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dimension,
        text_encoder_name=args.text_encoder,
        hidden_channels=args.HIDDEN_DIM,
        num_layers=args.encoder_layer,
        num_heads=args.encoder_head,
        num_relations=19
    )
    decoder = TransformerProgramDecoder(
        hidden_size=args.HIDDEN_DIM, program_length=args.program_length, 
        op_list=op_list, const_list=const_list,
        max_input_length=MAX_NODES,
        num_layers=args.decoder_layer,
        num_heads=args.decoder_head
    )
    model = IndoMPFGATe(encoder, decoder)
    
    # Load the trained weights from the checkpoint file
    print(f"Loading trained model checkpoint from: {args.model_checkpoint}")
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    
    # Handle both new dictionary-based checkpoints and old state_dict-only checkpoints
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model

def decode_program(program_ids, single_graph_data, op_list, const_list):
    """
    Converts a tensor of predicted token IDs back into a human-readable program string.
    """
    full_vocab = op_list + const_list
    reserved_token_size = len(full_vocab)
    id_to_token = {i: token for i, token in enumerate(full_vocab)}
    
    decoded_tokens = []
    for token_id_tensor in program_ids:
        token_id = token_id_tensor.item()
        if token_id == op_list.index('EOS'): break
        if token_id == 0 or token_id == op_list.index('GO'): continue

        if token_id < reserved_token_size:
            decoded_tokens.append(id_to_token.get(token_id, '[UNK]'))
        else:
            node_idx = token_id - reserved_token_size
            if node_idx < len(single_graph_data.tokens):
                pointed_word = single_graph_data.tokens[node_idx]
                pointed_loc = single_graph_data.node_locations[node_idx]
                decoded_tokens.append(f"ptr_to(word='{pointed_word}', loc={pointed_loc})")
            else:
                decoded_tokens.append(f"[POINTER_ERROR:{node_idx}]")
    return " ".join(decoded_tokens)

def test(args):
    """
    Evaluates the model on the test dataset to measure generalization performance.
    """
    print("\n--- Starting Testing ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- 2. Load Test Data ---
    op_list = ['GO', 'EOS', '(', ')', ',', "retrieve", "addition", "subtraction", "multiplication", "division", "exponential", "greater", "smaller", "equal", "sum", "average", "minimum", "maximum", "maximum_n", "minimum_n", "count", "count_if_equal", "count_if_less", "count_if_greater", "count_if_geq", "count_if_leq", "count_if_notequal", "sum_if_equal", "sum_if_less", "sum_if_greater", "sum_if_geq", "sum_if_leq", "sum_if_notequal", "filter_if_equal", "filter_if_less", "filter_if_greater", "filter_if_geq", "filter_if_leq", "filter_if_notequal", "trace_column", "trace_row", "unique"] + [f'#{i}' for i in range(1000)]
    const_list = [f'const_{i}' for i in range(1000)]

    # experiment id
    exp_id = timestamp_exp_start+"_"+args.exp_id if timestamp_exp_start!=args.exp_id else args.exp_id
    
    print(f"Loading and processing test data from: {args.test_file}")
    test_dataset = ProgramDataset(
        root=os.path.join(args.output_dir, 'test'),
        #root=os.path.join(args.output_dir, 'valid'),
        json_files=args.test_file,
        op_list=op_list, const_list=const_list, program_length=args.program_length,
        tokenizer=args.text_encoder
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # --- 1. Load Model ---
    model = load_model_for_inference(args, device)

    # --- 3. Evaluation Loop ---
    total_em = 0
    all_test_targets, all_test_predictions = [], []
    
    prediction_log_path = os.path.join(args.output_dir, exp_id+'test_predictions.txt')
    with open(prediction_log_path, 'w', encoding='utf-8') as log_file:
        log_file.write("Table UID | Question UID | Target Program | Predicted Program | Target Token ID | Predicted Token ID\n")
        log_file.write("="*120 + "\n")

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="[Testing]"):
                batch = batch.to(device)
                target_program_flat = batch.y
                target_program = target_program_flat.view(batch.num_graphs, -1)

                predicted_program = model.predict(batch, max_length=args.program_length)
                
                # --- Calculate Metrics ---
                is_match = (predicted_program == target_program).all(dim=-1)
                total_em += is_match.sum().item()

                targets_flat = target_program.view(-1)
                preds_flat = predicted_program.view(-1)
                non_padding_mask = targets_flat != 0
                all_test_targets.extend(targets_flat[non_padding_mask].cpu().numpy())
                all_test_predictions.extend(preds_flat[non_padding_mask].cpu().numpy())

                # --- Log Predictions ---
                graph_list = batch.to_data_list()
                for i in range(batch.num_graphs):
                    target_ids = target_program[i]
                    predicted_ids = predicted_program[i]
                    decoded_target = decode_program(target_program[i], graph_list[i], op_list, const_list)
                    decoded_predicted = decode_program(predicted_program[i], graph_list[i], op_list, const_list)
                    tbl_uid = graph_list[i].table_uid
                    qst_uid = graph_list[i].question_uid
                    log_file.write(f"{tbl_uid} | {qst_uid} | {decoded_target} | {decoded_predicted} | {str(target_ids.tolist())} | {str(predicted_ids.tolist())}\n")

    # --- 4. Report Final Metrics ---
    test_em_accuracy = total_em / len(test_dataset)
    test_token_accuracy = accuracy_score(all_test_targets, all_test_predictions)
    _, _, test_f1, _ = precision_recall_fscore_support(
        all_test_targets, all_test_predictions, average='macro', zero_division=0
    )

    print("\n--- Test Set Performance ---")
    print(f"  Exact Match (EM): {test_em_accuracy:.4f}")
    print(f"  Token Accuracy:   {test_token_accuracy:.4f}")
    print(f"  Macro F1-Score:   {test_f1:.4f}")
    print(f"Detailed predictions saved to: {prediction_log_path}")

def test_valid(args):
    """
    Evaluates the model on the test dataset to measure generalization performance.
    """
    print("\n--- Starting Testing ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- 2. Load Test Data ---
    op_list = ['GO', 'EOS', '(', ')', ',', "retrieve", "addition", "subtraction", "multiplication", "division", "exponential", "greater", "smaller", "equal", "sum", "average", "minimum", "maximum", "maximum_n", "minimum_n", "count", "count_if_equal", "count_if_less", "count_if_greater", "count_if_geq", "count_if_leq", "count_if_notequal", "sum_if_equal", "sum_if_less", "sum_if_greater", "sum_if_geq", "sum_if_leq", "sum_if_notequal", "filter_if_equal", "filter_if_less", "filter_if_greater", "filter_if_geq", "filter_if_leq", "filter_if_notequal", "trace_column", "trace_row", "unique"] + [f'#{i}' for i in range(1000)]
    const_list = [f'const_{i}' for i in range(1000)]
    
    # experiment id
    exp_id = timestamp_exp_start+"_"+args.exp_id if timestamp_exp_start!=args.exp_id else args.exp_id

    print(f"Loading and processing test data from: {args.test_file}")
    test_dataset = ProgramDataset(
        # root=os.path.join(args.output_dir, 'test'),
        root=os.path.join(args.output_dir, 'valid'),
        json_files=args.test_file,
        op_list=op_list, const_list=const_list, program_length=args.program_length,
        tokenizer=args.text_encoder
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # --- 1. Load Model ---
    model = load_model_for_inference(args, device)

    # --- 3. Evaluation Loop ---
    total_em = 0
    all_test_targets, all_test_predictions = [], []
    
    prediction_log_path = os.path.join(args.output_dir, exp_id+'validation_set_predictions.txt')
    with open(prediction_log_path, 'w', encoding='utf-8') as log_file:
        log_file.write("Table UID | Question UID | Target Program | Predicted Program | Target Token ID | Predicted Token ID\n")
        log_file.write("="*120 + "\n")

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="[Testing]"):
                batch = batch.to(device)
                target_program_flat = batch.y
                target_program = target_program_flat.view(batch.num_graphs, -1)

                predicted_program = model.predict(batch, max_length=args.program_length)
                
                # --- Calculate Metrics ---
                is_match = (predicted_program == target_program).all(dim=-1)
                total_em += is_match.sum().item()

                targets_flat = target_program.view(-1)
                preds_flat = predicted_program.view(-1)
                non_padding_mask = targets_flat != 0
                all_test_targets.extend(targets_flat[non_padding_mask].cpu().numpy())
                all_test_predictions.extend(preds_flat[non_padding_mask].cpu().numpy())

                # --- Log Predictions ---
                graph_list = batch.to_data_list()
                for i in range(batch.num_graphs):
                    target_ids = target_program[i]
                    predicted_ids = predicted_program[i]
                    decoded_target = decode_program(target_program[i], graph_list[i], op_list, const_list)
                    decoded_predicted = decode_program(predicted_program[i], graph_list[i], op_list, const_list)
                    tbl_uid = graph_list[i].table_uid
                    qst_uid = graph_list[i].question_uid
                    log_file.write(f"{tbl_uid} | {qst_uid} | {decoded_target} | {decoded_predicted} | {str(target_ids.tolist())} | {str(predicted_ids.tolist())}\n")

    # --- 4. Report Final Metrics ---
    test_em_accuracy = total_em / len(test_dataset)
    test_token_accuracy = accuracy_score(all_test_targets, all_test_predictions)
    _, _, test_f1, _ = precision_recall_fscore_support(
        all_test_targets, all_test_predictions, average='macro', zero_division=0
    )

    print("\n--- Test Set Performance ---")
    print(f"  Exact Match (EM): {test_em_accuracy:.4f}")
    print(f"  Token Accuracy:   {test_token_accuracy:.4f}")
    print(f"  Macro F1-Score:   {test_f1:.4f}")
    print(f"Detailed predictions saved to: {prediction_log_path}")

def predict(args):
    """
    Generates programs for a new, unseen JSON file.
    """
    print("\n--- Starting Prediction ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- 1. Load Model ---
    model = load_model_for_inference(args, device)
    
    # --- 2. Prepare Data ---
    op_list = ['GO', 'EOS', '(', ')', ',', "retrieve", "addition", "subtraction", "multiplication", "division", "exponential", "greater", "smaller", "equal", "sum", "average", "minimum", "maximum", "maximum_n", "minimum_n", "count", "count_if_equal", "count_if_less", "count_if_greater", "count_if_geq", "count_if_leq", "count_if_notequal", "sum_if_equal", "sum_if_less", "sum_if_greater", "sum_if_geq", "sum_if_leq", "sum_if_notequal", "filter_if_equal", "filter_if_less", "filter_if_greater", "filter_if_geq", "filter_if_leq", "filter_if_notequal", "trace_column", "trace_row", "unique"] + [f'#{i}' for i in range(1000)]
    const_list = [f'const_{i}' for i in range(1000)]

    # We need to format the new data so the ProgramDataset can process it.
    # The `process` method expects a JSON file with a specific structure.
    print(f"Reading new data from: {args.predict_file}")
    with open(args.predict_file) as f:
        new_data = json.load(f)

    # Add dummy fields that the processor expects but aren't in prediction data.
    for item in new_data:
        for q in item['questions']:
            q['uid'] = q.get('uid', str(uuid.uuid4())) # Generate a UID if none exists
            q['validity_status'] = "matched facts" # Assume valid for processing
            q['reasoning'] = {} # Dummy reasoning
            
    # Save this temporary, formatted data to a temp file
    temp_predict_file = os.path.join(args.output_dir, "temp_predict.json")
    with open(temp_predict_file, 'w') as f:
        json.dump(new_data, f)

    # Now, use the ProgramDataset to process this temporary file
    predict_dataset = ProgramDataset(
        root=os.path.join(args.output_dir, 'predict_cache'),
        json_files=temp_predict_file,
        op_list=op_list, const_list=const_list, program_length=args.program_length,
        tokenizer=args.text_encoder
    )
    predict_loader = DataLoader(predict_dataset, batch_size=args.batch_size, shuffle=False)

    # --- 3. Generation Loop ---
    all_results = []
    with torch.no_grad():
        for batch in tqdm(predict_loader, desc="[Predicting]"):
            batch = batch.to(device)
            predicted_program = model.predict(batch, max_length=args.program_length)
            
            graph_list = batch.to_data_list()
            for i in range(batch.num_graphs):
                decoded_predicted = decode_program(predicted_program[i], graph_list[i], op_list, const_list)
                result = {
                    "table_uid": graph_list[i].table_uid,
                    "question_uid": graph_list[i].question_uid,
                    "question": graph_list[i].question,
                    "predicted_program": decoded_predicted
                }
                all_results.append(result)

    # --- 4. Save Results ---
    results_path = os.path.join(args.output_dir, 'predictions.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
        
    print(f"\nPrediction complete. Results saved to: {results_path}")
    # Clean up the temporary file
    os.remove(temp_predict_file)

if __name__ == '__main__':
    # --- Command-Line Interface (CLI) Setup ---
    parser = argparse.ArgumentParser(description="Train or predict with the GraphProgramGenerator model.")
    
    # Shared arguments
    parser.add_argument("--mode", type=str, required=True, choices=['train', 'train-blaze', 'train-blaze-auto', 'predict', 'test','test-valid'], help="Run in 'train', 'test' or 'predict' mode.")
    parser.add_argument("--text_encoder", type=str, default="deepseek_qwen", help="Name of the pre-trained language model.")
    parser.add_argument("--output_dir", type=str, default="./output2", help="Directory to save models and logs.")
    
    # Training-specific arguments
    parser.add_argument("--train_file", type=str, default="data/dummy_dataset/raw/sample.json", help="Path to the training JSON file.")
    parser.add_argument("--valid_file", type=str, default="data/dummy_dataset/valid.json", help="Path to the validation JSON file.")
    parser.add_argument("--test_file", type=str, default="data/dummy_dataset/test.json", help="Path to the test JSON file.")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and validation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--program_length", type=int, default=1500, help="Max length for program sequences.")
    parser.add_argument("--HIDDEN_DIM", type=int, default=768, help="Hidden dimension in encoder/decoder layers.")
    parser.add_argument("--max_rows", type=int, default=1500, help="Max row for input table.")
    parser.add_argument("--max_cols", type=int, default=1500, help="Max col for input table.")
    
    # --- Argument to specify a checkpoint to resume from ---
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a model checkpoint to resume training from.")
    parser.add_argument("--resume_history", type=str, default=None, help="Path to a training_history.json file to resume logging from.")

    # Prediction-specific arguments
    parser.add_argument("--predict_file", type=str, help="Path to the new JSON file for prediction.")
    parser.add_argument("--model_checkpoint", type=str, default="./output/best_model.pt", help="Path to the trained model checkpoint.")

    parser.add_argument("--exp_id", type=str, default=timestamp_exp_start, help="Experiment Identification")

    # Add argument to set the number of head and layer in decoder/encoder
    parser.add_argument("--decoder_head", type=int, default=4, help="Number of attention head in decoder.")
    parser.add_argument("--decoder_layer", type=int, default=4, help="Number of Transformer decoder layer.")
    parser.add_argument("--encoder_head", type=int, default=4, help="Number of attention head in encoder.")
    parser.add_argument("--encoder_layer", type=int, default=4, help="Number of RGAT layer in encoder.")

    args = parser.parse_args()
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    torch.cuda.empty_cache()
    
    if args.mode == 'train':        
        train(args)

    elif args.mode == 'train-blaze':        
        train_blaze(args)

    elif args.mode == 'train-blaze-auto':        
        train_blaze_auto(args)

    elif args.mode=='test-valid':
        test_valid(args)

    elif args.mode=='test':
        test(args)

    elif args.mode == 'predict':
        if not args.predict_file:
            raise ValueError("Argument --predict_file is required for predict mode.")
        predict(args)