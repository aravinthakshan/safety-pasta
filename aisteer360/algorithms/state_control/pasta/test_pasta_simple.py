"""
Simple PASTA test script - visually compare outputs with and without PASTA.

Usage:
    python -m aisteer360.algorithms.state_control.pasta.test_pasta_simple
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    print("=" * 70)
    print("SIMPLE PASTA TEST")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = model.device
    print(f"Model loaded on {device}")
    
    # Test prompts with clear instructions
    test_prompts = [
        {
            "prompt": "Write exactly 3 sentences about cats. Each sentence must start with 'Cats'.",
            "instructions": ["Write exactly 3 sentences", "Each sentence must start with 'Cats'"],
        },
        {
            "prompt": "List 5 fruits. Do not use the letter 'a' in any fruit name.",
            "instructions": ["List 5 fruits", "Do not use the letter 'a'"],
        },
        {
            "prompt": "Explain what Python is in exactly 2 bullet points. Use markdown format.",
            "instructions": ["exactly 2 bullet points", "Use markdown format"],
        },
    ]
    
    # Generation settings
    gen_kwargs = {
        "max_new_tokens": 200,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    print("\n" + "=" * 70)
    print("BASELINE (No PASTA)")
    print("=" * 70)
    
    for i, test in enumerate(test_prompts):
        print(f"\n--- Test {i+1} ---")
        print(f"Prompt: {test['prompt']}")
        print(f"Instructions: {test['instructions']}")
        
        # Format with chat template
        messages = [{"role": "user", "content": test["prompt"]}]
        formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs
            )
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\nResponse:\n{response}")
    
    # Now test with PASTA
    print("\n" + "=" * 70)
    print("WITH PASTA (alpha=0.01, exclude mode)")
    print("=" * 70)
    
    from aisteer360.algorithms.state_control.pasta.control import PASTA
    from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
    
    # Create PASTA with aggressive settings
    pasta = PASTA(
        head_config=[8, 9, 10, 11, 12],  # Multiple layers
        alpha=0.01,  # Strong effect
        scale_position="exclude",
    )
    
    # Create pipeline
    pipeline = SteeringPipeline(
        model_name_or_path=None,
        controls=[pasta],
        lazy_init=True,
    )
    pipeline.model = model
    pipeline.tokenizer = tokenizer
    pipeline.device = device
    pipeline.steer()
    
    for i, test in enumerate(test_prompts):
        print(f"\n--- Test {i+1} ---")
        print(f"Prompt: {test['prompt']}")
        print(f"Instructions to emphasize: {test['instructions']}")
        
        # Format with chat template
        messages = [{"role": "user", "content": test["prompt"]}]
        formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        
        # Runtime kwargs with substrings
        runtime_kwargs = {"substrings": test["instructions"]}
        
        output_ids = pipeline.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            runtime_kwargs=runtime_kwargs,
            **gen_kwargs
        )
        
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\nResponse:\n{response}")
    
    # Test with INCLUDE mode and high alpha
    print("\n" + "=" * 70)
    print("WITH PASTA (alpha=10.0, include mode - BOOST instructions)")
    print("=" * 70)
    
    pasta_boost = PASTA(
        head_config=[8, 9, 10, 11, 12],
        alpha=10.0,  # Boost attention TO instructions
        scale_position="include",
    )
    
    pipeline2 = SteeringPipeline(
        model_name_or_path=None,
        controls=[pasta_boost],
        lazy_init=True,
    )
    pipeline2.model = model
    pipeline2.tokenizer = tokenizer
    pipeline2.device = device
    pipeline2.steer()
    
    for i, test in enumerate(test_prompts):
        print(f"\n--- Test {i+1} ---")
        print(f"Prompt: {test['prompt']}")
        
        messages = [{"role": "user", "content": test["prompt"]}]
        formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        runtime_kwargs = {"substrings": test["instructions"]}
        
        output_ids = pipeline2.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            runtime_kwargs=runtime_kwargs,
            **gen_kwargs
        )
        
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\nResponse:\n{response}")
    
    print("\n" + "=" * 70)
    print("DONE - Compare the outputs above!")
    print("=" * 70)


if __name__ == "__main__":
    main()
