"""
Fine-Tuning Example with LoRA
==============================
This script demonstrates how to fine-tune an LLM using LoRA (Low-Rank Adaptation).

Prerequisites:
    pip install transformers datasets peft accelerate torch bitsandbytes

Note: This is a template. For actual training, you'll need:
- GPU access (at least 16GB VRAM for 7B models with QLoRA)
- Training dataset in proper format
- Hugging Face account (for model access if gated)
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from trl import SFTTrainer  # Supervised Fine-Tuning Trainer

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "mistralai/Mistral-7B-v0.1"  # Or use "meta-llama/Llama-2-7b-hf"
OUTPUT_DIR = "./fine-tuned-model"
DATASET_NAME = "mlabonne/guanaco-llama2-1k"  # Sample dataset for demonstration

# LoRA Configuration
LORA_CONFIG = {
    "r": 16,                    # Rank of update matrices (typical: 8-64)
    "lora_alpha": 32,           # Scaling factor (typically 2x r)
    "target_modules": [         # Which layers to fine-tune
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM
}

# Training Configuration
TRAINING_ARGS = {
    "output_dir": OUTPUT_DIR,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "save_strategy": "epoch",
    "fp16": True,               # Use mixed precision
    "optim": "paged_adamw_8bit", # Memory-optimized optimizer
    "max_grad_norm": 0.3,
    "report_to": "none"         # Disable wandb for this example
}

# Quantization Configuration (for QLoRA)
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_use_double_quant": True,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_trainable_parameters(model):
    """Print the number of trainable parameters vs total parameters."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"Trainable parameters: {trainable_params:,d}")
    print(f"All parameters: {all_param:,d}")
    print(f"Trainable %: {100 * trainable_params / all_param:.2f}%")

def formatting_prompts_func(example):
    """Format dataset into instruction-following format."""
    text = f"### Instruction: {example['instruction']}\n\n### Input: {example['input']}\n\n### Output: {example['output']}"
    return {"text": text}

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def load_and_prepare_dataset():
    """Load and preprocess the training dataset."""
    print("Loading dataset...")
    
    # Load dataset from Hugging Face
    dataset = load_dataset(DATASET_NAME, split="train")
    
    # For demonstration, we're using a pre-formatted dataset
    # In production, you'd format your own data here
    
    # Split into train/test
    dataset = dataset.train_test_split(test_size=0.1)
    
    print(f"Training samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    print(f"Sample entry: {dataset['train'][0]}")
    
    return dataset

def load_base_model():
    """Load the base model with quantization enabled."""
    print(f"Loading model: {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        **QUANTIZATION_CONFIG,
        trust_remote_code=True
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def apply_lora(model):
    """Apply LoRA adapters to the model."""
    print("Applying LoRA adapters...")
    
    lora_config = LoraConfig(**LORA_CONFIG)
    
    model = get_peft_model(model, lora_config)
    
    print_trainable_parameters(model)
    
    return model

def train_model(model, tokenizer, dataset):
    """Train the model with SFTTrainer."""
    print("Starting training...")
    
    training_args = TrainingArguments(**TRAINING_ARGS)
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=512,
        packing=False,
    )
    
    # Train
    trainer.train()
    
    # Save the model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Model saved to {OUTPUT_DIR}")
    
    return trainer

def test_inference(model, tokenizer, prompt):
    """Test the fine-tuned model with a sample prompt."""
    print(f"\nTesting inference with prompt: {prompt}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text:\n{result}")
    
    return result

# ============================================================================
# ALTERNATIVE: CPU-ONLY VERSION (FOR DEMONSTRATION)
# ============================================================================

def cpu_only_demo():
    """
    Demonstrate the fine-tuning setup without requiring GPU.
    This is for educational purposes - actual training needs GPU.
    """
    print("="*60)
    print("CPU-Only Demonstration Mode")
    print("="*60)
    print("\nThis demo shows the fine-tuning pipeline structure.")
    print("Actual training requires GPU acceleration.\n")
    
    # Show what the configuration would look like
    print("Configuration Summary:")
    print(f"- Model: {MODEL_NAME}")
    print(f"- LoRA Rank: {LORA_CONFIG['r']}")
    print(f"- Target Modules: {len(LORA_CONFIG['target_modules'])} layers")
    print(f"- Training Epochs: {TRAINING_ARGS['num_train_epochs']}")
    print(f"- Batch Size: {TRAINING_ARGS['per_device_train_batch_size']}")
    print(f"- Learning Rate: {TRAINING_ARGS['learning_rate']}")
    
    # Calculate parameter efficiency
    print("\nParameter Efficiency with LoRA:")
    print("- Typical reduction: 10,000x fewer trainable parameters")
    print("- A 7B model might only train ~10M parameters instead of 7B")
    print("- Memory usage: ~12GB vs ~80GB for full fine-tuning")
    
    print("\nKey Steps in Production:")
    print("1. Prepare high-quality dataset (100-1000+ examples)")
    print("2. Choose appropriate base model")
    print("3. Configure LoRA hyperparameters")
    print("4. Train on GPU with monitoring")
    print("5. Evaluate on held-out test set")
    print("6. Merge LoRA weights or serve with adapter")
    print("7. Deploy to production")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("🚀 LLM Fine-Tuning with LoRA")
    print("="*60)
    
    # Check if GPU is available
    if not torch.cuda.is_available():
        print("\n⚠️  No GPU detected. Running in demonstration mode.")
        cpu_only_demo()
        return
    
    print(f"\nGPU Available: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    try:
        # Step 1: Load dataset
        dataset = load_and_prepare_dataset()
        
        # Step 2: Load base model
        model, tokenizer = load_base_model()
        
        # Step 3: Apply LoRA
        model = apply_lora(model)
        
        # Step 4: Train
        trainer = train_model(model, tokenizer, dataset)
        
        # Step 5: Test inference
        test_prompt = "### Instruction: Explain what machine learning is\n\n### Input: \n\n### Output: "
        test_inference(model, tokenizer, test_prompt)
        
        print("\n✅ Fine-tuning complete!")
        
    except Exception as e:
        print(f"\n❌ Error during fine-tuning: {str(e)}")
        print("\nTip: Make sure you have:")
        print("- Sufficient GPU memory (16GB+ recommended)")
        print("- Required packages installed")
        print("- Access to the model (some require approval)")

if __name__ == "__main__":
    main()
