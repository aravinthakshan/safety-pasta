"""
Circuit Tracker for Data-Driven PASTA Configuration using TransformerLens.

This module provides utilities to automatically learn optimal PASTA steering configurations
by analyzing attention circuits using causal intervention methods.

The approach uses TransformerLens for proper mechanistic interpretability:
1. Load model as HookedTransformer for activation access
2. Run activation patching to identify causal attention heads
3. Use attention knockout to measure head importance
4. Select heads that causally affect instruction-following
5. Construct PASTA config with head_config and alpha

Reference:
- TransformerLens: https://github.com/neelnanda-io/TransformerLens
- Inspired by "Locating and Editing Factual Associations in GPT" (Meng et al.)
- Causal tracing methodology from mechanistic interpretability research
"""

from __future__ import annotations

from typing import Sequence, Callable
from functools import partial

import torch
import numpy as np

# TransformerLens imports
try:
    from transformer_lens import HookedTransformer, ActivationCache
    from transformer_lens.hook_points import HookPoint
    TRANSFORMER_LENS_AVAILABLE = True
except ImportError:
    TRANSFORMER_LENS_AVAILABLE = False
    HookedTransformer = None
    ActivationCache = None
    HookPoint = None


class CircuitTracker:
    """Tracks attention circuits using TransformerLens causal interventions.
    
    This class uses activation patching and attention knockout to identify
    attention heads that are causally responsible for instruction-following
    behavior. These heads can then be used to configure PASTA.
    
    Args:
        model_name: HuggingFace model name to load via TransformerLens.
        device: Device to run computations on.
        
    Example:
        >>> tracker = CircuitTracker("Qwen/Qwen2.5-1.5B-Instruct")
        >>> head_scores = tracker.run_activation_patching(clean_prompts, corrupted_prompts)
        >>> head_config, alpha = tracker.get_top_heads(head_scores, top_k=10)
        >>> pasta = PASTA(head_config=head_config, alpha=alpha)
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        if not TRANSFORMER_LENS_AVAILABLE:
            raise ImportError(
                "TransformerLens is required for causal circuit analysis. "
                "Install with: pip install transformer_lens"
            )
        
        print(f"Loading model {model_name} with TransformerLens...")
        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            dtype=dtype,
        )
        self.model.eval()
        
        self.device = device
        self.model_name = model_name
        self.num_layers = self.model.cfg.n_layers
        self.num_heads = self.model.cfg.n_heads
        self.d_head = self.model.cfg.d_head
        
        print(f"  Model loaded: {self.num_layers} layers, {self.num_heads} heads")
        
        # Storage for analysis results
        self.head_importance_scores: np.ndarray | None = None
    
    def _get_logit_diff(
        self,
        logits: torch.Tensor,
        target_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute average log probability of target tokens.
        
        This serves as a proxy for "how well the model is doing" - higher is better.
        """
        # logits: (batch, seq, vocab)
        # target_tokens: (batch, seq)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Gather log probs for target tokens
        # target_tokens for next-token prediction is input shifted by 1
        target_log_probs = torch.gather(
            log_probs[:, :-1, :],  # (batch, seq-1, vocab)
            dim=-1,
            index=target_tokens[:, 1:].unsqueeze(-1)  # (batch, seq-1, 1)
        ).squeeze(-1)  # (batch, seq-1)
        
        if attention_mask is not None:
            # Only consider non-padding positions
            mask = attention_mask[:, 1:]  # shift to match target
            target_log_probs = target_log_probs * mask
            return target_log_probs.sum() / mask.sum()
        else:
            return target_log_probs.mean()
    
    def run_attention_knockout(
        self,
        prompts: Sequence[str],
        batch_size: int = 4,
        use_chat_template: bool = True,
    ) -> np.ndarray:
        """Run attention knockout to measure head importance.
        
        For each attention head, we zero out its output and measure how much
        the model's loss increases. Heads that cause large loss increases
        are more important.
        
        Args:
            prompts: List of prompts to evaluate on.
            batch_size: Batch size for processing.
            use_chat_template: Whether to apply chat template.
            
        Returns:
            Array of shape (num_layers, num_heads) with importance scores.
            Higher scores = more important heads.
        """
        print(f"\nRunning attention knockout analysis...")
        print(f"  Testing {self.num_layers * self.num_heads} heads")
        
        # Format prompts
        if use_chat_template:
            formatted = []
            for p in prompts:
                try:
                    # TransformerLens models may have different chat templates
                    formatted.append(f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n")
                except:
                    formatted.append(p)
            prompts = formatted
        
        # Get baseline loss
        baseline_losses = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            tokens = self.model.to_tokens(batch, prepend_bos=True)
            
            with torch.no_grad():
                logits = self.model(tokens)
                loss = self._get_logit_diff(logits, tokens)
                baseline_losses.append(loss.item())
        
        baseline_loss = np.mean(baseline_losses)
        print(f"  Baseline loss: {baseline_loss:.4f}")
        
        # Test each head
        head_importance = np.zeros((self.num_layers, self.num_heads))
        
        for layer in range(self.num_layers):
            print(f"  Layer {layer + 1}/{self.num_layers}...", end=" ", flush=True)
            
            for head in range(self.num_heads):
                # Create hook to zero out this head's output
                def knockout_hook(
                    activation: torch.Tensor,
                    hook: HookPoint,
                    head_idx: int,
                ) -> torch.Tensor:
                    # activation shape: (batch, seq, num_heads, d_head)
                    activation[:, :, head_idx, :] = 0
                    return activation
                
                hook_fn = partial(knockout_hook, head_idx=head)
                hook_name = f"blocks.{layer}.attn.hook_z"
                
                # Run with knockout
                knockout_losses = []
                for i in range(0, len(prompts), batch_size):
                    batch = prompts[i:i + batch_size]
                    tokens = self.model.to_tokens(batch, prepend_bos=True)
                    
                    with torch.no_grad():
                        logits = self.model.run_with_hooks(
                            tokens,
                            fwd_hooks=[(hook_name, hook_fn)],
                        )
                        loss = self._get_logit_diff(logits, tokens)
                        knockout_losses.append(loss.item())
                
                knockout_loss = np.mean(knockout_losses)
                
                # Importance = how much loss increased when we knocked out this head
                # Negative because lower log prob = higher loss = more important
                head_importance[layer, head] = baseline_loss - knockout_loss
            
            print(f"done")
        
        self.head_importance_scores = head_importance
        return head_importance
    
    def run_activation_patching(
        self,
        clean_prompts: Sequence[str],
        corrupted_prompts: Sequence[str],
        batch_size: int = 4,
        metric: str = "logit_diff",
    ) -> np.ndarray:
        """Run activation patching to find causal heads.
        
        This is the gold standard for causal circuit analysis:
        1. Run model on "clean" examples (successful instruction following)
        2. Run model on "corrupted" examples (failed instruction following)
        3. For each head, patch its activation from clean → corrupted
        4. Measure how much this restores the clean behavior
        
        Heads that restore behavior when patched are causally important.
        
        Args:
            clean_prompts: Prompts where model succeeds.
            corrupted_prompts: Prompts where model fails (same length).
            batch_size: Batch size for processing.
            metric: Metric to use ("logit_diff" or "loss").
            
        Returns:
            Array of shape (num_layers, num_heads) with patching scores.
            Higher = more causal importance.
        """
        if len(clean_prompts) != len(corrupted_prompts):
            raise ValueError("clean_prompts and corrupted_prompts must have same length")
        
        print(f"\nRunning activation patching analysis...")
        print(f"  {len(clean_prompts)} prompt pairs")
        print(f"  Testing {self.num_layers * self.num_heads} heads")
        
        patching_scores = np.zeros((self.num_layers, self.num_heads))
        
        for batch_start in range(0, len(clean_prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(clean_prompts))
            clean_batch = clean_prompts[batch_start:batch_end]
            corrupted_batch = corrupted_prompts[batch_start:batch_end]
            
            # Tokenize
            clean_tokens = self.model.to_tokens(clean_batch, prepend_bos=True)
            corrupted_tokens = self.model.to_tokens(corrupted_batch, prepend_bos=True)
            
            # Get clean activations (cache them)
            with torch.no_grad():
                _, clean_cache = self.model.run_with_cache(clean_tokens)
            
            # Get corrupted baseline
            with torch.no_grad():
                corrupted_logits = self.model(corrupted_tokens)
                corrupted_metric = self._get_logit_diff(corrupted_logits, corrupted_tokens).item()
            
            # Get clean baseline  
            with torch.no_grad():
                clean_logits = self.model(clean_tokens)
                clean_metric = self._get_logit_diff(clean_logits, clean_tokens).item()
            
            total_effect = clean_metric - corrupted_metric
            
            # Patch each head
            for layer in range(self.num_layers):
                for head in range(self.num_heads):
                    def patching_hook(
                        activation: torch.Tensor,
                        hook: HookPoint,
                        clean_activation: torch.Tensor,
                        head_idx: int,
                    ) -> torch.Tensor:
                        # Patch just this head's output
                        activation[:, :, head_idx, :] = clean_activation[:, :, head_idx, :]
                        return activation
                    
                    # Get the clean activation for this layer
                    clean_z = clean_cache[f"blocks.{layer}.attn.hook_z"]
                    hook_fn = partial(patching_hook, clean_activation=clean_z, head_idx=head)
                    hook_name = f"blocks.{layer}.attn.hook_z"
                    
                    with torch.no_grad():
                        patched_logits = self.model.run_with_hooks(
                            corrupted_tokens,
                            fwd_hooks=[(hook_name, hook_fn)],
                        )
                        patched_metric = self._get_logit_diff(patched_logits, corrupted_tokens).item()
                    
                    # How much did patching this head restore clean behavior?
                    restoration = patched_metric - corrupted_metric
                    if total_effect != 0:
                        patching_scores[layer, head] += restoration / total_effect
            
            # Clear cache
            del clean_cache
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Average over batches
        num_batches = (len(clean_prompts) + batch_size - 1) // batch_size
        patching_scores /= num_batches
        
        self.head_importance_scores = patching_scores
        return patching_scores
    
    def run_attention_pattern_analysis(
        self,
        success_prompts: Sequence[str],
        failure_prompts: Sequence[str],
        batch_size: int = 4,
    ) -> np.ndarray:
        """Analyze attention patterns to find differentially active heads.
        
        This is a faster (but less rigorous) alternative to activation patching.
        Compares attention entropy between success and failure prompts.
        
        Args:
            success_prompts: Prompts where model succeeds.
            failure_prompts: Prompts where model fails.
            batch_size: Batch size for processing.
            
        Returns:
            Array of shape (num_layers, num_heads) with delta scores.
        """
        print(f"\nRunning attention pattern analysis...")
        
        def get_attention_stats(prompts):
            all_entropy = []
            
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                tokens = self.model.to_tokens(batch, prepend_bos=True)
                
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(tokens)
                
                batch_entropy = np.zeros((len(batch), self.num_layers, self.num_heads))
                
                for layer in range(self.num_layers):
                    # attention pattern: (batch, num_heads, seq, seq)
                    pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
                    
                    for head in range(self.num_heads):
                        head_pattern = pattern[:, head, :, :]  # (batch, seq, seq)
                        
                        # Compute entropy for each example
                        for b in range(len(batch)):
                            attn = head_pattern[b]
                            # Entropy: -sum(p * log(p))
                            log_attn = torch.log(attn + 1e-10)
                            entropy = -torch.sum(attn * log_attn, dim=-1).mean()
                            batch_entropy[b, layer, head] = entropy.item()
                
                all_entropy.append(batch_entropy)
                del cache
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return np.concatenate(all_entropy, axis=0)
        
        print(f"  Analyzing {len(success_prompts)} success prompts...")
        success_entropy = get_attention_stats(success_prompts)
        
        print(f"  Analyzing {len(failure_prompts)} failure prompts...")
        failure_entropy = get_attention_stats(failure_prompts)
        
        # Compute delta: higher entropy in failures = more diffuse attention
        success_mean = success_entropy.mean(axis=0)
        failure_mean = failure_entropy.mean(axis=0)
        delta = failure_mean - success_mean
        
        self.head_importance_scores = delta
        return delta
    
    def get_top_heads(
        self,
        scores: np.ndarray | None = None,
        top_k: int = 10,
        alpha_scale: float = 0.05,
        alpha_min: float = 0.001,
        alpha_max: float = 0.1,
    ) -> tuple[dict[int, list[int]], float]:
        """Get top-K heads from importance scores.
        
        Args:
            scores: Head importance scores (num_layers, num_heads).
                   If None, uses self.head_importance_scores.
            top_k: Number of top heads to select.
            alpha_scale: Base alpha scaling factor.
            alpha_min: Minimum alpha value.
            alpha_max: Maximum alpha value.
            
        Returns:
            Tuple of (head_config, alpha) for PASTA.
        """
        if scores is None:
            scores = self.head_importance_scores
        if scores is None:
            raise ValueError("No scores available. Run an analysis method first.")
        
        # Handle NaN
        scores = np.nan_to_num(scores, nan=0.0)
        
        # Flatten and get top-K
        flat_scores = scores.flatten()
        top_indices = np.argsort(np.abs(flat_scores))[-top_k:][::-1]
        
        # Build head_config
        head_config: dict[int, list[int]] = {}
        selected_scores = []
        
        for flat_idx in top_indices:
            layer = int(flat_idx // self.num_heads)
            head = int(flat_idx % self.num_heads)
            score = float(flat_scores[flat_idx])
            
            if layer not in head_config:
                head_config[layer] = []
            head_config[layer].append(head)
            selected_scores.append(abs(score))
        
        # Sort heads within each layer
        for layer in head_config:
            head_config[layer] = sorted(head_config[layer])
        
        # Compute alpha
        if selected_scores and max(selected_scores) > 0:
            normalized = np.array(selected_scores) / max(selected_scores)
            mean_norm = float(normalized.mean())
            alpha = float(np.clip(alpha_scale * (1 + mean_norm), alpha_min, alpha_max))
        else:
            alpha = float(alpha_scale)
        
        print(f"\n=== Circuit Analysis Summary ===")
        print(f"Selected {top_k} heads across {len(head_config)} layers")
        print(f"Head config: {head_config}")
        print(f"Computed alpha: {alpha:.4f}")
        print(f"================================\n")
        
        return head_config, alpha
    
    def visualize_scores(
        self,
        scores: np.ndarray | None = None,
        title: str = "Head Importance Scores",
    ):
        """Visualize head importance scores as a heatmap.
        
        Requires matplotlib to be installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for visualization")
            return
        
        if scores is None:
            scores = self.head_importance_scores
        if scores is None:
            raise ValueError("No scores to visualize")
        
        plt.figure(figsize=(12, 8))
        plt.imshow(scores, aspect='auto', cmap='RdBu_r')
        plt.colorbar(label='Importance Score')
        plt.xlabel('Head')
        plt.ylabel('Layer')
        plt.title(title)
        plt.tight_layout()
        plt.savefig('head_importance.png', dpi=150)
        print("Saved visualization to head_importance.png")


def create_circuit_pasta(
    model_name: str,
    prompts: Sequence[str],
    success_mask: Sequence[bool],
    top_k: int = 10,
    alpha_scale: float = 0.05,
    scale_position: str = "exclude",
    method: str = "knockout",
    **kwargs,
):
    """Convenience function to create a circuit-informed PASTA instance.
    
    Args:
        model_name: HuggingFace model name.
        prompts: List of prompts to analyze.
        success_mask: Boolean list indicating success/failure.
        top_k: Number of top heads to select.
        alpha_scale: Base alpha scaling factor.
        scale_position: PASTA scale position.
        method: Analysis method ("knockout", "patching", or "pattern").
        **kwargs: Additional arguments for the analysis method.
        
    Returns:
        Configured PASTA instance.
    """
    from aisteer360.algorithms.state_control.pasta.control import PASTA
    
    tracker = CircuitTracker(model_name)
    
    success_prompts = [p for p, s in zip(prompts, success_mask) if s]
    failure_prompts = [p for p, s in zip(prompts, success_mask) if not s]
    
    if method == "knockout":
        scores = tracker.run_attention_knockout(prompts, **kwargs)
    elif method == "patching":
        # For patching, we need paired examples
        min_len = min(len(success_prompts), len(failure_prompts))
        scores = tracker.run_activation_patching(
            success_prompts[:min_len],
            failure_prompts[:min_len],
            **kwargs,
        )
    elif method == "pattern":
        scores = tracker.run_attention_pattern_analysis(
            success_prompts, failure_prompts, **kwargs
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    head_config, alpha = tracker.get_top_heads(scores, top_k=top_k, alpha_scale=alpha_scale)
    
    return PASTA(
        head_config=head_config,
        alpha=alpha,
        scale_position=scale_position,
    )


# =============================================================================
# RUNNABLE EVALUATION SCRIPT
# =============================================================================

def run_circuit_pasta_evaluation(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    num_samples: int = 50,
    top_k: int = 10,
    alpha_scale: float = 0.05,
    batch_size: int = 8,
    max_new_tokens: int = 1024,
    save_dir: str = "./circuit_pasta_results",
    analysis_method: str = "knockout",
):
    """
    Run complete circuit-informed PASTA evaluation on Split-IFEval.
    
    Uses TransformerLens for causal circuit analysis.
    """
    import gc
    import json
    import time
    from pathlib import Path
    
    import nltk
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, logging
    
    from aisteer360.algorithms.state_control.pasta.control import PASTA
    from aisteer360.evaluation.use_cases.instruction_following import InstructionFollowing
    from aisteer360.evaluation.metrics.custom.instruction_following.strict_instruction import StrictInstruction
    from aisteer360.evaluation.benchmark import Benchmark
    
    logging.set_verbosity_error()
    
    # Download NLTK data
    print("Downloading required NLTK data...")
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    print("NLTK data ready.")
    
    def log_time(start_time, step_name):
        elapsed = time.time() - start_time
        print(f"  [time] {step_name} took {elapsed:.1f}s")
        return time.time()
    
    total_start = time.time()
    
    print("=" * 60)
    print("CIRCUIT-INFORMED PASTA EVALUATION (TransformerLens)")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Samples: {num_samples}")
    print(f"Top-K heads: {top_k}")
    print(f"Alpha scale: {alpha_scale}")
    print(f"Analysis method: {analysis_method}")
    print("=" * 60)
    
    # Step 1: Load dataset
    step_start = time.time()
    print("\n[1/6] Loading Split-IFEval dataset...")
    dataset = load_dataset("ibm-research/Split-IFEval", split="train")
    evaluation_data = dataset.to_list()[:num_samples]
    prompts = [d["prompt"] for d in evaluation_data]
    print(f"  Loaded {len(evaluation_data)} examples")
    step_start = log_time(step_start, "Dataset loading")
    
    # Step 2: Run baseline evaluation
    print("\n[2/6] Running baseline evaluation...")
    instruction_following = InstructionFollowing(
        evaluation_data=evaluation_data,
        evaluation_metrics=[StrictInstruction()],
    )
    
    baseline_benchmark = Benchmark(
        use_case=instruction_following,
        base_model_name_or_path=model_name,
        steering_pipelines={"baseline": []},
        gen_kwargs={"max_new_tokens": max_new_tokens, "do_sample": False},
        hf_model_kwargs={"attn_implementation": "eager", "torch_dtype": torch.float16},
        batch_size=batch_size,
    )
    baseline_profiles = baseline_benchmark.run()
    step_start = log_time(step_start, "Baseline evaluation")
    
    baseline_results = baseline_profiles["baseline"][0]["evaluations"]["StrictInstruction"]
    follow_all = baseline_results["follow_all_instructions"]
    
    print(f"Baseline prompt accuracy: {baseline_results['strict_prompt_accuracy']:.2%}")
    print(f"Baseline instruction accuracy: {baseline_results['strict_instruction_accuracy']:.2%}")
    print(f"Successes: {sum(follow_all)}, Failures: {len(follow_all) - sum(follow_all)}")
    
    # Clean up baseline model before loading TransformerLens
    del baseline_benchmark
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Step 3: Circuit analysis with TransformerLens
    print("\n[3/6] Running circuit analysis with TransformerLens...")
    
    tracker = CircuitTracker(model_name)
    
    success_prompts = [p for p, s in zip(prompts, follow_all) if s]
    failure_prompts = [p for p, s in zip(prompts, follow_all) if not s]
    
    if analysis_method == "knockout":
        scores = tracker.run_attention_knockout(prompts, batch_size=batch_size)
    elif analysis_method == "patching":
        min_len = min(len(success_prompts), len(failure_prompts))
        if min_len == 0:
            print("Warning: Need both successes and failures for patching. Using knockout.")
            scores = tracker.run_attention_knockout(prompts, batch_size=batch_size)
        else:
            scores = tracker.run_activation_patching(
                success_prompts[:min_len],
                failure_prompts[:min_len],
                batch_size=batch_size,
            )
    else:  # pattern
        if len(success_prompts) == 0 or len(failure_prompts) == 0:
            print("Warning: Need both successes and failures. Using knockout.")
            scores = tracker.run_attention_knockout(prompts, batch_size=batch_size)
        else:
            scores = tracker.run_attention_pattern_analysis(
                success_prompts, failure_prompts, batch_size=batch_size
            )
    
    step_start = log_time(step_start, "Circuit analysis")
    
    # Step 4: Get top heads
    print("\n[4/6] Selecting top circuit heads...")
    head_config, alpha = tracker.get_top_heads(scores, top_k=top_k, alpha_scale=alpha_scale)
    
    # Show top heads
    flat_scores = scores.flatten()
    top_indices = np.argsort(np.abs(flat_scores))[-5:][::-1]
    print("\nTop 5 circuit heads:")
    for idx in top_indices:
        layer = idx // tracker.num_heads
        head = idx % tracker.num_heads
        score = flat_scores[idx]
        print(f"  Layer {layer}, Head {head}: score = {score:.4f}")
    
    # Clean up TransformerLens model
    del tracker
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Step 5: Run comparison
    print("\n[5/6] Running evaluation with circuit PASTA...")
    
    circuit_pasta = PASTA(
        head_config=head_config,
        alpha=alpha,
        scale_position="exclude",
    )
    
    manual_pasta = PASTA(
        head_config=[8, 9],
        alpha=0.01,
        scale_position="exclude",
    )
    
    instruction_following_2 = InstructionFollowing(
        evaluation_data=evaluation_data,
        evaluation_metrics=[StrictInstruction()],
    )
    
    comparison_benchmark = Benchmark(
        use_case=instruction_following_2,
        base_model_name_or_path=model_name,
        steering_pipelines={
            "baseline": [],
            "manual_pasta": [manual_pasta],
            "circuit_pasta": [circuit_pasta],
        },
        runtime_overrides={"PASTA": {"substrings": "instructions"}},
        gen_kwargs={"max_new_tokens": max_new_tokens, "do_sample": False},
        hf_model_kwargs={"attn_implementation": "eager", "torch_dtype": torch.float16},
        batch_size=batch_size,
    )
    
    comparison_profiles = comparison_benchmark.run()
    step_start = log_time(step_start, "Comparison benchmark")
    
    # Step 6: Report results
    print("\n[6/6] Results Summary")
    print("=" * 60)
    
    results_table = []
    for method_name in ["baseline", "manual_pasta", "circuit_pasta"]:
        scores_dict = comparison_profiles[method_name][0]["evaluations"]["StrictInstruction"]
        results_table.append({
            "method": method_name,
            "prompt_accuracy": scores_dict["strict_prompt_accuracy"],
            "instruction_accuracy": scores_dict["strict_instruction_accuracy"],
        })
        print(f"\n{method_name.upper()}:")
        print(f"  Prompt Accuracy:      {scores_dict['strict_prompt_accuracy']:.2%}")
        print(f"  Instruction Accuracy: {scores_dict['strict_instruction_accuracy']:.2%}")
    
    baseline_prompt = results_table[0]["prompt_accuracy"]
    baseline_instr = results_table[0]["instruction_accuracy"]
    
    print("\n" + "-" * 60)
    print("IMPROVEMENTS OVER BASELINE:")
    for result in results_table[1:]:
        print(f"\n{result['method'].upper()}:")
        print(f"  Prompt:      {result['prompt_accuracy'] - baseline_prompt:+.2%}")
        print(f"  Instruction: {result['instruction_accuracy'] - baseline_instr:+.2%}")
    
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
            "analysis_method": analysis_method,
        },
        "circuit_config": {
            "head_config": {str(k): v for k, v in head_config.items()},
            "alpha": alpha,
        },
        "results": results_table,
    }
    
    with open(save_path / "circuit_pasta_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {save_path / 'circuit_pasta_results.json'}")
    
    total_elapsed = time.time() - total_start
    print(f"\n[COMPLETE] Total time: {total_elapsed/60:.1f} minutes ({total_elapsed:.0f}s)")
    
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run circuit-informed PASTA evaluation using TransformerLens",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", "-m", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--num-samples", "-n", type=int, default=50)
    parser.add_argument("--top-k", "-k", type=int, default=10)
    parser.add_argument("--alpha-scale", "-a", type=float, default=0.05)
    parser.add_argument("--batch-size", "-b", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--save-dir", "-o", type=str, default="./circuit_pasta_results")
    parser.add_argument(
        "--method", type=str, default="knockout",
        choices=["knockout", "patching", "pattern"],
        help="Circuit analysis method: knockout (fastest), patching (gold standard), pattern (simple)"
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
        analysis_method=args.method,
    )
