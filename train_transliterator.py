import os
os.environ["TORCH_USE_CUDA_DSA"] = "1" # Enable CUDA DSA
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

from datasets import load_dataset
from transformers import Trainer, Seq2SeqTrainingArguments, EncoderDecoderModel, EncoderDecoderConfig, BertConfig, DataCollatorForSeq2Seq, PreTrainedTokenizerFast
from utils import lowercase_text

if __name__ == "__main__":
    
    # Learning parameters
    learning_rate = 3e-5
    batch_size = 128
    n_epochs = 120
    weight_decay = 0.005
    save_total_limit = 5
    fp16 = False
    bf16 = False
    warmup_ratio = 0.02
    gradient_accumulation_steps = 2
    test_size = 0.01  # (1% of the data, around 700 samples)

    # Transformer parameters
    d_model = 1024
    nhead = 8
    num_encoder_layers = 8
    num_decoder_layers = 8

    # Data paths
    DATA_PATH = 'atlasia/ATAM'

    output_dir="atlasia/Transliteration-Moroccan-Darija"

    # Set the source and target (labels)
    src = 'darija_arabizi'
    tgt = 'darija_arabic'

    # Load training dataset from Hugging Face datasets
    dataset = load_dataset(DATA_PATH, split="train")

    # Transform the column to lowercase
    print(f'[INFO] Transforming the {src} column to lowercase')
    dataset = dataset.map(lambda example: {src: lowercase_text(example[src])}, batched=True)
    
    # Load the pretrained BPE tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
    tokenizer.add_special_tokens({'pad_token': '<pad>', 'bos_token': '<bos>', 'eos_token': '<eos>', 'unk_token': '<unk>'})

    dataset = dataset.train_test_split(test_size=test_size)

    # Define the encoder configuration
    encoder_config = BertConfig(
        hidden_size=d_model,
        num_attention_heads=nhead,
        num_hidden_layers=num_encoder_layers,
    )

    # Define the decoder configuration
    decoder_config = BertConfig(
        hidden_size=d_model,
        num_attention_heads=nhead,
        num_hidden_layers=num_decoder_layers,
    )

    # Instantiate the model configuration
    config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
    config.decoder_start_token_id = 1 #tokenizer.token_to_id("<bos>")
    config.pad_token_id = 0 # tokenizer.token_to_id("<pad>")

    # Instantiate the model using the configuration
    model = EncoderDecoderModel(config=config)

    print(f'The model has {model.num_parameters()} parameters')

    # Define the preprocess function to tokenize inputs and targets
    def preprocess_function(examples, tokenizer, max_length=128):
        inputs = examples[src]
        targets = examples[tgt]

        # Tokenize inputs and targets
        input_tokenized = tokenizer(inputs, padding=True, truncation=True)
        target_tokenized = tokenizer(targets, padding=True, truncation=True)

        # Pad tokenized sequences to the max length
        input_tokenized = tokenizer.pad(input_tokenized, max_length=max_length, padding='max_length', return_tensors='pt')
        target_tokenized = tokenizer.pad(target_tokenized, max_length=max_length, padding='max_length', return_tensors='pt')

        model_inputs = {
            "input_ids": input_tokenized["input_ids"],
            "attention_mask": input_tokenized["attention_mask"],
            "labels": target_tokenized["input_ids"],
        }

        return model_inputs

    # Instantiate data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    tokenized_data = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # Define the training arguments
    training_args = Seq2SeqTrainingArguments(
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        num_train_epochs=n_epochs,
        weight_decay=weight_decay,
        save_total_limit=save_total_limit,
        bf16=bf16,
        fp16=fp16,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_dir="logs",
        report_to="tensorboard",
        output_dir=output_dir,
        push_to_hub=True,
    )

    # Instantiate the Trainer class
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        data_collator=data_collator,
    )

    # Start training
    trainer.train()
    
    trainer.push_to_hub()
