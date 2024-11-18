import os
os.environ["TORCH_USE_CUDA_DSA"] = "1" # Enable CUDA DSA
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers
from tokenizers.pre_tokenizers import Whitespace
from utils import lowercase_text


if __name__ == "__main__":
    
    # Data paths
    DATA_PATH = 'atlasia/ATAM'
    
    # Output tokenizer path
    TOKENIZER_PATH = 'tokenizer.json'

    # Set the source and target (labels)
    src = 'darija_arabizi'
    tgt = 'darija_arabic'

    # Load training dataset from Hugging Face datasets
    dataset = load_dataset(DATA_PATH, split="train")

    # Transform the source data to lowercase. As this model takes Arabizi input and produce in Ary.
    print(f'[INFO] Transforming the {src} column to lowercase')
    dataset = dataset.map(lambda example: {src: lowercase_text(example[src])}, batched=True)

    # Train Byte-Pair Encoding (BPE) tokenizer
    tokenizer_bpe = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer_bpe.pre_tokenizer = Whitespace()
    trainer_bpe = trainers.BpeTrainer(special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"], vocab_size=30000, min_frequency=1)
    tokenizer_bpe.train_from_iterator(dataset[src] + dataset[tgt], trainer=trainer_bpe)

    # Save the trained tokenizer
    tokenizer_bpe.save(TOKENIZER_PATH)

   