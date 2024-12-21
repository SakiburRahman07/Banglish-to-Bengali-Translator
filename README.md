# Banglish to Bengali Transliteration using Transformers

This README provides detailed instructions for training a machine learning model to transliterate Banglish (Romanized Bengali) text to Bengali using the Hugging Face Transformers library. The steps include loading the dataset, preprocessing the data, selecting a model, training the model, and testing it with sample inputs.

## Project Output

Check out this video to see the project in action:

[Check out our demo video](https://www.youtube.com/watch?v=_e2Zw_VVpAA)



## Prerequisites

Ensure you have the following libraries installed:

```bash
pip install datasets transformers sentencepiece torch --upgrade
```

## Steps

### 1. Configure PyTorch CUDA Allocation (Optional)

Set an environment variable to optimize GPU memory usage:

```python
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
```

### 2. Import Libraries

Import the necessary libraries:

```python
from datasets import load_dataset
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import torch
import gc
```

### 3. Define a Function to Clear GPU Memory

```python
def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
    print(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")
```

Clear memory before starting:

```python
clear_memory()
```

### 4. Load the Dataset

Load the Banglish to Bengali dataset:

```python
dataset = load_dataset("SKNahin/bengali-transliteration-data")
print("Dataset Structure:", dataset)
print("Features:", dataset['train'].features)
print("First Example:", dataset['train'][0])
```

### 5. Split the Dataset

Split the dataset into training and validation subsets:

```python
train_test_split = dataset['train'].train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split['train']
val_dataset = train_test_split['test']
```

Limit the dataset size for resource management:

```python
train_size = min(1000, len(train_dataset))
val_size = min(200, len(val_dataset))
train_dataset = train_dataset.select(range(train_size))
val_dataset = val_dataset.select(range(val_size))
```

### 6. Filter Empty Examples

Remove rows with empty inputs or outputs:

```python
def filter_empty_examples(example):
    return len(example["rm"].strip()) > 0 and len(example["bn"].strip()) > 0

train_dataset = train_dataset.filter(filter_empty_examples)
val_dataset = val_dataset.filter(filter_empty_examples)

print(f"Train dataset size after filtering: {len(train_dataset)}")
print(f"Validation dataset size after filtering: {len(val_dataset)}")
```

### 7. Load the Tokenizer and Model

Initialize the tokenizer and model:

```python
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
```

Enable gradient checkpointing for memory optimization:

```python
model.gradient_checkpointing_enable()
```

### 8. Preprocess the Data

Define a preprocessing function:

```python
def preprocess_function(examples):
    inputs = examples["rm"]  # Banglish
    targets = examples["bn"]  # Bengali

    inputs = [f"translate en to ben: {text}" for text in inputs]
    model_inputs = tokenizer(inputs, max_length=64, truncation=True, padding=True)

    labels = tokenizer(targets, max_length=64, truncation=True, padding=True)
    labels["input_ids"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_ids]
        for labels_ids in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["rm", "bn"])
val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=["rm", "bn"])
```

### 9. Create a Data Collator

Create a data collator for dynamic padding:

```python
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, label_pad_token_id=-100, padding=True
)
```

### 10. Configure Training Arguments

Set training parameters:

```python
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=3,
    predict_with_generate=True,
    fp16=True,
    logging_dir='./logs',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    report_to="none",
    gradient_accumulation_steps=8,
)
```

### 11. Initialize the Trainer

Create a trainer instance:

```python
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
```

### 12. Train the Model

Clear GPU memory before training:

```python
clear_memory()
```

Train the model:

```python
trainer.train()
```

Save the trained model and tokenizer:

```python
trainer.save_model("./banglish_to_bangla_model")
tokenizer.save_pretrained("./banglish_to_bangla_model")
```

### 13. Test the Model

Define a translation function:

```python
def translate_text(input_text, model, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    with torch.no_grad():
        input_text = f"translate en to ben: {input_text}"
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=64,
            truncation=True,
            padding=True
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_length=64,
            num_beams=5,
            early_stopping=True
        )

        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text
```

Test the model with sample inputs:

```python
sample_sentences = [
    "ami bhalo achi",
    "tumi kemon acho",
    "tara bazar jachhe",
    "amar naam Iqbal",
    "ami school jabo",
    "tader bari kothay?",
]

for sentence in sample_sentences:
    translated = translate_text(sentence, model, tokenizer)
    print(f"Banglish: {sentence}\nBengali: {translated}\n")
