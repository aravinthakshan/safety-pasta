"""
Circuit Tracker for Data-Driven PASTA Configuration.

This module provides utilities to automatically learn optimal PASTA steering configurations
by analyzing attention patterns correlated with instruction-following failures.

The approach uses contrastive attention statistics (NOT full causal patching):
1. Capture attention patterns during forward passes on prompts
2. Separate examples into instruction-success vs instruction-failure groups
3. Compute per-(layer, head) statistics: mean attention entropy or norm
4. Compute delta scores: Δ = failure_mean − success_mean
5. Select top-K heads with highest positive Δ as "instruction-failure circuits"
6. Construct PASTA config with head_config and alpha proportional to normalized Δ

Reference:
- Inspired by circuit analysis techniques from mechanistic interpretability
- Uses contrastive statistics rather than activation patching for efficiency
"""

from __future__ import annotations

from typing import Sequence

import torch
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class CircuitTracker:
    """Tracks attention circuits correlated with instruction-following failures.
    
    This class captures attention patterns during forward passes and identifies
    attention heads that are differentially active during instruction-following
    failures vs successes. These heads can then be used to configure PASTA
    for improved instruction following.
    
    Args:
        model: HuggingFace causal language model with attention output support.
        tokenizer: Tokenizer for the model.
        device: Device to run computations on. Defaults to model's device.
        
    Example:
        >>> tracker = CircuitTracker(model, tokenizer)
        >>> tracker.capture_batch(prompts)
        >>> head_config, alpha = tracker.analyze(success_mask, top_k=10)
        >>> pasta = PASTA(head_config=head_config, alpha=alpha)
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device | str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        
        # Model architecture info
        self.num_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads
        
        # Storage for attention statistics per example
        # Shape after capture: (num_examples, num_layers, num_heads)
        self.attention_entropy: list[np.ndarray] = []
        self.attention_norm: list[np.ndarray] = []
        
        # Ensure tokenizer has padding configured
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"
    
    def capture_batch(
        self,
        prompts: Sequence[str],
        batch_size: int = 8,
        max_length: int = 2048,
        use_chat_template: bool = True,
    ) -> None:
        """Capture attention patterns for a batch of prompts.
        
        Runs forward passes on each prompt and stores per-(layer, head) attention
        statistics (entropy and L2 norm) for later analysis.
        
        Args:
            prompts: List of prompt strings to analyze.
            batch_size: Number of prompts to process at once.
            max_length: Maximum sequence length for tokenization.
            use_chat_template: Whether to apply chat template to prompts.
        """
        self.model.eval()
        
        # Apply chat template if available and requested
        if use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            formatted_prompts = []
            for prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                formatted_prompts.append(formatted)
            prompts = formatted_prompts
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)
            
            # Forward pass with attention outputs
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_attentions=True,
                    return_dict=True,
                )
            
            # Process attentions: tuple of (batch, num_heads, seq_len, seq_len) per layer
            attentions = outputs.attentions  # tuple of length num_layers
            
            # Compute statistics for each example in batch
            for b in range(len(batch_prompts)):
                # Get attention mask for this example to identify valid positions
                mask = inputs["attention_mask"][b]
                valid_len = mask.sum().item()
                
                example_entropy = np.zeros((self.num_layers, self.num_heads))
                example_norm = np.zeros((self.num_layers, self.num_heads))
                
                for layer_idx, layer_attn in enumerate(attentions):
                    # layer_attn shape: (batch, num_heads, seq_len, seq_len)
                    attn = layer_attn[b]  # (num_heads, seq_len, seq_len)
                    
                    for head_idx in range(self.num_heads):
                        head_attn = attn[head_idx, :valid_len, :valid_len]  # (valid_len, valid_len)
                        
                        # Compute attention entropy (averaged over query positions)
                        # Entropy = -sum(p * log(p)) for each query position
                        entropy = self._compute_entropy(head_attn)
                        example_entropy[layer_idx, head_idx] = entropy
                        
                        # Compute attention L2 norm (Frobenius norm of attention matrix)
                        norm = torch.norm(head_attn, p='fro').item()
                        example_norm[layer_idx, head_idx] = norm
                
                self.attention_entropy.append(example_entropy)
                self.attention_norm.append(example_norm)
            
            # Free memory
            del outputs, attentions
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _compute_entropy(self, attn_matrix: torch.Tensor, eps: float = 1e-10) -> float:
        """Compute mean attention entropy across query positions.
        
        Entropy measures how "spread out" the attention is. High entropy means
        attention is distributed across many positions; low entropy means focused.
        
        Args:
            attn_matrix: Attention weights of shape (query_len, key_len).
            eps: Small constant for numerical stability.
            
        Returns:
            Mean entropy across all query positions.
        """
        # attn_matrix rows should sum to 1 (softmax output)
        # Entropy for each row: -sum(p * log(p))
        log_attn = torch.log(attn_matrix + eps)
        entropy_per_query = -torch.sum(attn_matrix * log_attn, dim=-1)  # (query_len,)
        return entropy_per_query.mean().item()
    
    def analyze(
        self,
        success_mask: Sequence[bool],
        top_k: int = 10,
        alpha_scale: float = 0.05,
        alpha_min: float = 0.001,
        alpha_max: float = 0.1,
        metric: str = "entropy",
    ) -> tuple[dict[int, list[int]], float | dict[int, list[float]]]:
        """Analyze captured attention patterns to derive PASTA configuration.
        
        Computes contrastive statistics between success and failure groups:
        - For each (layer, head), compute mean statistic for successes vs failures
        - Delta = failure_mean - success_mean
        - Positive delta indicates heads more active during failures
        - Select top-K heads with highest positive delta
        
        Args:
            success_mask: Boolean list where True = instruction-following success.
            top_k: Number of top heads to select for PASTA config.
            alpha_scale: Base scaling factor for alpha values.
            alpha_min: Minimum alpha value (clipping).
            alpha_max: Maximum alpha value (clipping).
            metric: Which metric to use - "entropy" or "norm".
            
        Returns:
            Tuple of (head_config, alpha):
            - head_config: Dict mapping layer index to list of head indices
            - alpha: Either a single float (mean alpha) or dict with per-head alphas
            
        Raises:
            ValueError: If no attention patterns have been captured.
        """
        if not self.attention_entropy:
            raise ValueError("No attention patterns captured. Call capture_batch() first.")
        
        if len(success_mask) != len(self.attention_entropy):
            raise ValueError(
                f"success_mask length ({len(success_mask)}) doesn't match "
                f"captured examples ({len(self.attention_entropy)})"
            )
        
        # Select metric
        if metric == "entropy":
            stats = np.array(self.attention_entropy)  # (num_examples, num_layers, num_heads)
        elif metric == "norm":
            stats = np.array(self.attention_norm)
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'entropy' or 'norm'.")
        
        success_mask = np.array(success_mask)
        
        # Separate into success and failure groups
        success_stats = stats[success_mask]
        failure_stats = stats[~success_mask]
        
        if len(success_stats) == 0:
            print("Warning: No successful examples. Using all examples as baseline.")
            success_stats = stats
        if len(failure_stats) == 0:
            print("Warning: No failed examples. Cannot compute meaningful delta.")
            # Return default config (first few layers, all heads)
            return {0: list(range(self.num_heads)), 1: list(range(self.num_heads))}, alpha_scale
        
        # Compute mean statistics per (layer, head)
        success_mean = success_stats.mean(axis=0)  # (num_layers, num_heads)
        failure_mean = failure_stats.mean(axis=0)  # (num_layers, num_heads)
        
        # Compute delta: positive means higher activity during failures
        # For entropy: higher entropy during failures = more diffuse attention = less focused
        # These heads might benefit from PASTA steering to refocus attention
        delta = failure_mean - success_mean  # (num_layers, num_heads)
        
        # Flatten and get top-K indices
        flat_delta = delta.flatten()
        top_k_indices = np.argsort(flat_delta)[-top_k:][::-1]  # Descending order
        
        # Convert flat indices to (layer, head) pairs
        selected_heads: list[tuple[int, int, float]] = []
        for flat_idx in top_k_indices:
            layer_idx = flat_idx // self.num_heads
            head_idx = flat_idx % self.num_heads
            delta_value = flat_delta[flat_idx]
            
            # Only include heads with positive delta (more active during failures)
            if delta_value > 0:
                selected_heads.append((layer_idx, head_idx, delta_value))
        
        if not selected_heads:
            print("Warning: No heads with positive delta found. Using top heads by absolute value.")
            top_k_indices = np.argsort(np.abs(flat_delta))[-top_k:][::-1]
            for flat_idx in top_k_indices:
                layer_idx = flat_idx // self.num_heads
                head_idx = flat_idx % self.num_heads
                delta_value = np.abs(flat_delta[flat_idx])
                selected_heads.append((layer_idx, head_idx, delta_value))
        
        # Build head_config dict: layer -> list of heads
        head_config: dict[int, list[int]] = {}
        for layer_idx, head_idx, _ in selected_heads:
            if layer_idx not in head_config:
                head_config[layer_idx] = []
            if head_idx not in head_config[layer_idx]:
                head_config[layer_idx][head_idx] = head_idx
                head_config[layer_idx] = list(set(head_config[layer_idx]) | {head_idx})
        
        # Sort heads within each layer
        for layer_idx in head_config:
            head_config[layer_idx] = sorted(head_config[layer_idx])
        
        # Compute alpha: proportional to normalized delta
        delta_values = np.array([d for _, _, d in selected_heads])
        if delta_values.max() > 0:
            # Normalize to [0, 1] range, then scale
            normalized_delta = delta_values / delta_values.max()
            mean_normalized = normalized_delta.mean()
            alpha = float(np.clip(alpha_scale * (1 + mean_normalized), alpha_min, alpha_max))
        else:
            alpha = alpha_scale
        
        # Print analysis summary
        print(f"\n=== Circuit Analysis Summary ===")
        print(f"Total examples: {len(stats)}")
        print(f"Successes: {success_mask.sum()}, Failures: {(~success_mask).sum()}")
        print(f"Selected {len(selected_heads)} heads across {len(head_config)} layers")
        print(f"Head config: {head_config}")
        print(f"Computed alpha: {alpha:.4f}")
        print(f"================================\n")
        
        return head_config, alpha
    
    def get_detailed_analysis(
        self,
        success_mask: Sequence[bool],
        metric: str = "entropy",
    ) -> dict:
        """Get detailed per-head analysis for inspection.
        
        Returns comprehensive statistics for debugging and visualization.
        
        Args:
            success_mask: Boolean list where True = instruction-following success.
            metric: Which metric to analyze - "entropy" or "norm".
            
        Returns:
            Dict containing:
            - delta_matrix: (num_layers, num_heads) array of delta values
            - success_mean: Mean statistics for successful examples
            - failure_mean: Mean statistics for failed examples
            - top_heads: List of (layer, head, delta) tuples sorted by delta
        """
        if metric == "entropy":
            stats = np.array(self.attention_entropy)
        else:
            stats = np.array(self.attention_norm)
        
        success_mask = np.array(success_mask)
        
        success_mean = stats[success_mask].mean(axis=0) if success_mask.any() else stats.mean(axis=0)
        failure_mean = stats[~success_mask].mean(axis=0) if (~success_mask).any() else stats.mean(axis=0)
        delta = failure_mean - success_mean
        
        # Get all heads sorted by delta
        top_heads = []
        for layer_idx in range(self.num_layers):
            for head_idx in range(self.num_heads):
                top_heads.append((layer_idx, head_idx, delta[layer_idx, head_idx]))
        top_heads.sort(key=lambda x: x[2], reverse=True)
        
        return {
            "delta_matrix": delta,
            "success_mean": success_mean,
            "failure_mean": failure_mean,
            "top_heads": top_heads,
            "num_examples": len(stats),
            "num_successes": int(success_mask.sum()),
            "num_failures": int((~success_mask).sum()),
        }
    
    def reset(self) -> None:
        """Clear all captured attention patterns."""
        self.attention_entropy.clear()
        self.attention_norm.clear()


def create_circuit_pasta(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    success_mask: Sequence[bool],
    top_k: int = 10,
    alpha_scale: float = 0.05,
    scale_position: str = "exclude",
    batch_size: int = 8,
    **tracker_kwargs,
):
    """Convenience function to create a circuit-informed PASTA instance.
    
    Combines CircuitTracker analysis with PASTA instantiation in one call.
    
    Args:
        model: HuggingFace model to analyze.
        tokenizer: Tokenizer for the model.
        prompts: List of prompts to analyze attention patterns on.
        success_mask: Boolean list indicating instruction-following success/failure.
        top_k: Number of top circuit heads to select.
        alpha_scale: Base alpha scaling factor.
        scale_position: PASTA scale position ("include", "exclude", or "generation").
        batch_size: Batch size for attention capture.
        **tracker_kwargs: Additional arguments passed to CircuitTracker.analyze().
        
    Returns:
        Configured PASTA instance with circuit-derived head_config and alpha.
        
    Example:
        >>> circuit_pasta = create_circuit_pasta(
        ...     model, tokenizer, prompts, follow_all_instructions,
        ...     top_k=10, alpha_scale=0.05
        ... )
        >>> pipeline = SteeringPipeline(controls=[circuit_pasta], ...)
    """
    from aisteer360.algorithms.state_control.pasta.control import PASTA
    
    # Create tracker and capture patterns
    tracker = CircuitTracker(model, tokenizer)
    tracker.capture_batch(prompts, batch_size=batch_size)
    
    # Analyze and get config
    head_config, alpha = tracker.analyze(
        success_mask=success_mask,
        top_k=top_k,
        alpha_scale=alpha_scale,
        **tracker_kwargs,
    )
    
    # Create and return PASTA instance
    return PASTA(
        head_config=head_config,
        alpha=alpha,
        scale_position=scale_position,
    )


# =============================================================================
# RUNNABLE EVALUATION SCRIPT
# =============================================================================
# Usage: python -m aisteer360.algorithms.state_control.pasta.circuit_tracker
# Or:    python circuit_tracker.py
# =============================================================================

def run_circuit_pasta_evaluation(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    num_samples: int = 50,
    top_k: int = 10,
    alpha_scale: float = 0.05,
    batch_size: int = 8,
    max_new_tokens: int = 1024,
    save_dir: str = "./circuit_pasta_results",
):
    """
    Run complete circuit-informed PASTA evaluation on Split-IFEval.
    
    This function:
    1. Loads the Split-IFEval dataset
    2. Runs baseline evaluation to get instruction-following success/failure
    3. Captures attention patterns during forward passes
    4. Analyzes circuits to find heads correlated with failures
    5. Creates circuit-informed PASTA configuration
    6. Re-runs evaluation with circuit PASTA
    7. Compares results with baseline and manual PASTA
    
    Args:
        model_name: HuggingFace model ID or path.
        num_samples: Number of examples to evaluate.
        top_k: Number of top circuit heads to select for PASTA.
        alpha_scale: Base alpha scaling factor.
        batch_size: Batch size for generation and attention capture.
        max_new_tokens: Maximum tokens to generate.
        save_dir: Directory to save results.
    """
    import gc
    import json
    from pathlib import Path
    
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, logging
    
    from aisteer360.algorithms.state_control.pasta.control import PASTA
    from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
    from aisteer360.evaluation.use_cases.instruction_following import InstructionFollowing
    from aisteer360.evaluation.metrics.custom.instruction_following.strict_instruction import StrictInstruction
    from aisteer360.evaluation.benchmark import Benchmark
    
    logging.set_verbosity_error()
    
    print("=" * 60)
    print("CIRCUIT-INFORMED PASTA EVALUATION")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Samples: {num_samples}")
    print(f"Top-K heads: {top_k}")
    print(f"Alpha scale: {alpha_scale}")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Step 1: Load dataset
    # -------------------------------------------------------------------------
    print("\n[1/6] Loading Split-IFEval dataset...")
    dataset = load_dataset("ibm-research/Split-IFEval", split="train")
    evaluation_data = dataset.to_list()[:num_samples]
    prompts = [d["prompt"] for d in evaluation_data]
    print(f"Loaded {len(evaluation_data)} examples")
    
    # -------------------------------------------------------------------------
    # Step 2: Create use case and run baseline
    # -------------------------------------------------------------------------
    print("\n[2/6] Running baseline evaluation...")
    instruction_following = InstructionFollowing(
        evaluation_data=evaluation_data,
        evaluation_metrics=[StrictInstruction()],
    )
    
    # Load model for baseline
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        attn_implementation="eager",  # Required for attention outputs
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Run baseline benchmark
    baseline_benchmark = Benchmark(
        use_case=instruction_following,
        base_model_name_or_path=model_name,
        steering_pipelines={"baseline": []},
        gen_kwargs={
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        },
        hf_model_kwargs={
            "attn_implementation": "eager",
            "torch_dtype": torch.float16,
        },
        batch_size=batch_size,
    )
    baseline_profiles = baseline_benchmark.run()
    
    # Extract success/failure mask
    baseline_results = baseline_profiles["baseline"][0]["evaluations"]["StrictInstruction"]
    follow_all_instructions = baseline_results["follow_all_instructions"]
    
    print(f"Baseline prompt accuracy: {baseline_results['strict_prompt_accuracy']:.2%}")
    print(f"Baseline instruction accuracy: {baseline_results['strict_instruction_accuracy']:.2%}")
    print(f"Successes: {sum(follow_all_instructions)}, Failures: {len(follow_all_instructions) - sum(follow_all_instructions)}")
    
    # -------------------------------------------------------------------------
    # Step 3: Capture attention patterns
    # -------------------------------------------------------------------------
    print("\n[3/6] Capturing attention patterns...")
    
    # Use the loaded model from benchmark
    tracker = CircuitTracker(
        model=baseline_benchmark._base_model,
        tokenizer=baseline_benchmark._base_tokenizer,
    )
    tracker.capture_batch(prompts, batch_size=batch_size)
    print(f"Captured attention patterns for {len(prompts)} prompts")
    
    # -------------------------------------------------------------------------
    # Step 4: Analyze circuits
    # -------------------------------------------------------------------------
    print("\n[4/6] Analyzing circuits...")
    head_config, alpha = tracker.analyze(
        success_mask=follow_all_instructions,
        top_k=top_k,
        alpha_scale=alpha_scale,
    )
    
    # Get detailed analysis for reporting
    detailed = tracker.get_detailed_analysis(follow_all_instructions)
    print(f"\nTop 5 circuit heads (by delta):")
    for layer, head, delta in detailed["top_heads"][:5]:
        print(f"  Layer {layer}, Head {head}: Δ = {delta:.4f}")
    
    # -------------------------------------------------------------------------
    # Step 5: Create steering pipelines and re-run
    # -------------------------------------------------------------------------
    print("\n[5/6] Running evaluation with circuit PASTA...")
    
    # Create circuit PASTA
    circuit_pasta = PASTA(
        head_config=head_config,
        alpha=alpha,
        scale_position="exclude",
    )
    
    # Also create manual PASTA for comparison (using layers 8,9 as in notebook)
    manual_pasta = PASTA(
        head_config=[8, 9],
        alpha=0.01,
        scale_position="exclude",
    )
    
    # Re-create use case (to reset state)
    instruction_following_2 = InstructionFollowing(
        evaluation_data=evaluation_data,
        evaluation_metrics=[StrictInstruction()],
    )
    
    # Run comparison benchmark
    comparison_benchmark = Benchmark(
        use_case=instruction_following_2,
        base_model_name_or_path=model_name,
        steering_pipelines={
            "baseline": [],
            "manual_pasta": [manual_pasta],
            "circuit_pasta": [circuit_pasta],
        },
        runtime_overrides={
            "PASTA": {"substrings": "instructions"},
        },
        gen_kwargs={
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "output_attentions": True,
        },
        hf_model_kwargs={
            "attn_implementation": "eager",
            "torch_dtype": torch.float16,
        },
        batch_size=batch_size,
    )
    
    comparison_profiles = comparison_benchmark.run()
    
    # -------------------------------------------------------------------------
    # Step 6: Report results
    # -------------------------------------------------------------------------
    print("\n[6/6] Results Summary")
    print("=" * 60)
    
    results_table = []
    for method_name in ["baseline", "manual_pasta", "circuit_pasta"]:
        scores = comparison_profiles[method_name][0]["evaluations"]["StrictInstruction"]
        results_table.append({
            "method": method_name,
            "prompt_accuracy": scores["strict_prompt_accuracy"],
            "instruction_accuracy": scores["strict_instruction_accuracy"],
        })
        print(f"\n{method_name.upper()}:")
        print(f"  Prompt Accuracy:      {scores['strict_prompt_accuracy']:.2%}")
        print(f"  Instruction Accuracy: {scores['strict_instruction_accuracy']:.2%}")
    
    # Compute improvements
    baseline_prompt = results_table[0]["prompt_accuracy"]
    baseline_instr = results_table[0]["instruction_accuracy"]
    
    print("\n" + "-" * 60)
    print("IMPROVEMENTS OVER BASELINE:")
    for result in results_table[1:]:
        prompt_diff = result["prompt_accuracy"] - baseline_prompt
        instr_diff = result["instruction_accuracy"] - baseline_instr
        print(f"\n{result['method'].upper()}:")
        print(f"  Prompt:      {prompt_diff:+.2%}")
        print(f"  Instruction: {instr_diff:+.2%}")
    
    print("\n" + "=" * 60)
    
    # Save results
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        "config": {
            "model": model_name,
            "num_samples": num_samples,
            "top_k": top_k,
            "alpha_scale": alpha_scale,
        },
        "circuit_config": {
            "head_config": {str(k): v for k, v in head_config.items()},
            "alpha": alpha,
        },
        "results": results_table,
        "detailed_analysis": {
            "top_10_heads": [
                {"layer": int(l), "head": int(h), "delta": float(d)}
                for l, h, d in detailed["top_heads"][:10]
            ],
            "num_successes": detailed["num_successes"],
            "num_failures": detailed["num_failures"],
        }
    }
    
    with open(save_path / "circuit_pasta_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {save_path / 'circuit_pasta_results.json'}")
    
    # Cleanup
    del tracker, model, baseline_benchmark, comparison_benchmark
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run circuit-informed PASTA evaluation on Split-IFEval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=50,
        help="Number of evaluation samples",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=10,
        help="Number of top circuit heads to select",
    )
    parser.add_argument(
        "--alpha-scale", "-a",
        type=float,
        default=0.05,
        help="Base alpha scaling factor for PASTA",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=8,
        help="Batch size for generation and attention capture",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--save-dir", "-o",
        type=str,
        default="./circuit_pasta_results",
        help="Directory to save results",
    )
    
    args = parser.parse_args()
    
    results = run_circuit_pasta_evaluation(
        model_name=args.model,
        num_samples=args.num_samples,
        top_k=args.top_k,
        alpha_scale=args.alpha_scale,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        save_dir=args.save_dir,
    )
