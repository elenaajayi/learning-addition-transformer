"""
Comprehensive evaluation script for the Addition Transformer.

This module provides tools for evaluating a trained model on various test sets,
including in-distribution tests, length generalization, and distribution shift tests.
It also supports interactive testing mode.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from config import TaskConfig, DataConfig, ModelConfig, EvalConfig
from model import AdditionTransformer, Tokenizer
from data_generator import AdditionDataGenerator


class Evaluator:
    """
    Evaluator for the trained addition transformer model.
    
    Provides comprehensive evaluation across multiple test sets and
    supports interactive testing for manual verification.
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the evaluator with a trained model.
        
        Args:
            model_path: Path to the model checkpoint file
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize configurations
        self.task_config = TaskConfig()
        self.model_config = ModelConfig(self.task_config)
        
        # Create tokenizer
        self.tokenizer = Tokenizer(self.task_config)
        
        # Maximum output length for generation (answer can be up to k+1 digits + EOS)
        self.max_output_len = self.task_config.K_DIGITS + 2
        
        # Initialize model
        self.model = AdditionTransformer(self.model_config, self.task_config)
        
        # Load checkpoint
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load state dict (handle both formats)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from {model_path} on {self.device}")
        
        # Store checkpoint info if available
        self.checkpoint_info = {}
        if 'epoch' in checkpoint:
            self.checkpoint_info['epoch'] = checkpoint['epoch']
        if 'val_loss' in checkpoint:
            self.checkpoint_info['val_loss'] = checkpoint['val_loss']
        if 'val_accuracy' in checkpoint:
            self.checkpoint_info['val_accuracy'] = checkpoint['val_accuracy']
    
    def _extract_answer(self, generated_tokens: List[int], input_len: int) -> str:
        """
        Extract the answer portion from generated tokens.
        
        Args:
            generated_tokens: Full list of generated token IDs
            input_len: Length of the input (to skip)
            
        Returns:
            Decoded answer string (without special tokens)
        """
        # Get tokens after the input
        answer_tokens = generated_tokens[input_len:]
        
        # Remove special tokens (PAD, BOS, EOS)
        clean_tokens = []
        for tok in answer_tokens:
            token_str = self.tokenizer._idx_to_token.get(tok, "")
            if token_str not in self.task_config.SPECIAL_TOKENS:
                clean_tokens.append(token_str)
            elif token_str == "EOS":
                break  # Stop at EOS
        
        return "".join(clean_tokens)
    
    def predict(self, input_str: str) -> str:
        """
        Generate prediction for a single input.
        
        Args:
            input_str: Problem string like "123+456="
            
        Returns:
            Predicted answer string
        """
        # Prepare input: BOS + tokenized input
        input_ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(input_str)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        input_len = len(input_ids)
        
        with torch.no_grad():
            # Generate output
            output_ids = self.model.generate(
                input_tensor,
                max_new_tokens=self.max_output_len,
                greedy=True
            )
        
        # Extract and decode answer
        generated_tokens = output_ids[0].tolist()
        predicted = self._extract_answer(generated_tokens, input_len)
        
        return predicted
    
    def evaluate_dataset(
        self, 
        dataset_path: str, 
        description: str = "Test Set"
    ) -> Dict:
        """
        Evaluate model on a dataset from a JSON file.
        
        Args:
            dataset_path: Path to JSON file containing test examples
            description: Description for progress bar
            
        Returns:
            Dictionary with evaluation metrics and sample predictions
        """
        # Load dataset
        with open(dataset_path, 'r') as f:
            examples = json.load(f)
        
        # Initialize counters
        total = 0
        correct = 0
        
        # Carry-specific counters
        carry_total = 0
        carry_correct = 0
        no_carry_total = 0
        no_carry_correct = 0
        
        # Sample collections (for inspection)
        sample_correct = []
        sample_incorrect = []
        
        # Evaluate each example
        for example in tqdm(examples, desc=description):
            # Extract input and expected output
            input_str = example["input"]  # e.g., "123+456="
            expected = example["output"]   # e.g., "579"
            
            # Generate prediction
            predicted = self.predict(input_str)
            
            # Check correctness
            is_correct = (predicted.strip() == expected.strip())
            
            # Update counters
            total += 1
            if is_correct:
                correct += 1
            
            # Track carry/no-carry accuracy if info available
            if "requires_carry" in example:
                if example["requires_carry"]:
                    carry_total += 1
                    if is_correct:
                        carry_correct += 1
                else:
                    no_carry_total += 1
                    if is_correct:
                        no_carry_correct += 1
            
            # Collect samples (limit to 5 each)
            sample_data = {
                "input": input_str,
                "expected": expected,
                "predicted": predicted
            }
            
            if is_correct and len(sample_correct) < 5:
                sample_correct.append(sample_data)
            elif not is_correct and len(sample_incorrect) < 5:
                sample_incorrect.append(sample_data)
        
        # Calculate accuracies (handle division by zero)
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        carry_accuracy = 100.0 * carry_correct / carry_total if carry_total > 0 else None
        no_carry_accuracy = 100.0 * no_carry_correct / no_carry_total if no_carry_total > 0 else None
        
        # Return results
        return {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "carry_total": carry_total,
            "carry_correct": carry_correct,
            "carry_accuracy": carry_accuracy,
            "no_carry_total": no_carry_total,
            "no_carry_correct": no_carry_correct,
            "no_carry_accuracy": no_carry_accuracy,
            "sample_correct": sample_correct,
            "sample_incorrect": sample_incorrect
        }
    
    def run_full_evaluation(self) -> Dict:
        """
        Run evaluation on all available test sets.
        
        Returns:
            Dictionary with results for each test set
        """
        # Define test sets: (filepath, key_name, description)
        test_sets = [
            ("data/test.json", "in_distribution", "In-Distribution (k=3)"),
            ("data/test_length_k4.json", "length_gen_k4", "Length Gen (k=4)"),
            ("data/test_length_k5.json", "length_gen_k5", "Length Gen (k=5)"),
            ("data/test_shift_many_nines.json", "shift_nines", "Dist Shift (Many 9s)"),
            ("data/test_shift_small_numbers.json", "shift_small", "Dist Shift (Small Nums)")
        ]
        
        results = {}
        
        print("\n" + "=" * 60)
        print("RUNNING FULL EVALUATION")
        print("=" * 60 + "\n")
        
        for filepath, key, description in test_sets:
            if Path(filepath).exists():
                print(f"\nEvaluating: {description}")
                results[key] = self.evaluate_dataset(filepath, description=description)
            else:
                if filepath == "data/test.json":
                    raise FileNotFoundError(
                        f"Required test set {filepath} not found. "
                        "Please run data_generator.py first."
                    )
                else:
                    print(f"Skipping {description}: {filepath} not found")
        
        return results
    
    def save_results(
        self, 
        results: Dict, 
        filepath: str = "results/evaluation_results.json"
    ):
        """
        Save evaluation results to a JSON file.
        
        Args:
            results: Dictionary of evaluation results
            filepath: Output file path
        """
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        output = {
            "checkpoint_info": self.checkpoint_info,
            "device": self.device,
            "results": results
        }
        
        # Write results
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to {filepath}")
    
    def print_results_table(self, results: Dict):
        """
        Print evaluation results in a formatted table.
        
        Args:
            results: Dictionary of evaluation results
        """
        # Column widths
        name_w = 28
        total_w = 8
        acc_w = 10
        carry_w = 12
        no_carry_w = 12
        
        # Table characters
        h_line = "─"
        v_line = "│"
        
        print("\n")
        
        # Top border
        print("┌" + h_line * name_w + "┬" + h_line * total_w + "┬" + 
              h_line * acc_w + "┬" + h_line * carry_w + "┬" + h_line * no_carry_w + "┐")
        
        # Header
        print(f"{v_line}{'Test Set':<{name_w}}{v_line}{'Total':^{total_w}}{v_line}"
              f"{'Accuracy':^{acc_w}}{v_line}{'Carry Acc':^{carry_w}}{v_line}"
              f"{'No-Carry':^{no_carry_w}}{v_line}")
        
        # Header separator
        print("├" + h_line * name_w + "┼" + h_line * total_w + "┼" + 
              h_line * acc_w + "┼" + h_line * carry_w + "┼" + h_line * no_carry_w + "┤")
        
        # Data rows
        display_names = {
            "in_distribution": "In-Distribution (k=3)",
            "length_gen_k4": "Length Gen (k=4)",
            "length_gen_k5": "Length Gen (k=5)",
            "shift_nines": "Dist Shift (Many 9s)",
            "shift_small": "Dist Shift (Small Nums)"
        }
        
        for key, data in results.items():
            name = display_names.get(key, key)
            total = str(data['total'])
            acc = f"{data['accuracy']:.1f}%"
            carry_acc = f"{data['carry_accuracy']:.1f}%" if data['carry_accuracy'] is not None else "N/A"
            no_carry_acc = f"{data['no_carry_accuracy']:.1f}%" if data['no_carry_accuracy'] is not None else "N/A"
            
            print(f"{v_line}{name:<{name_w}}{v_line}{total:^{total_w}}{v_line}"
                  f"{acc:^{acc_w}}{v_line}{carry_acc:^{carry_w}}{v_line}"
                  f"{no_carry_acc:^{no_carry_w}}{v_line}")
        
        # Bottom border
        print("└" + h_line * name_w + "┴" + h_line * total_w + "┴" + 
              h_line * acc_w + "┴" + h_line * carry_w + "┴" + h_line * no_carry_w + "┘")
        
        # Print some sample predictions
        print("\n" + "=" * 60)
        print("SAMPLE PREDICTIONS")
        print("=" * 60)
        
        for key, data in results.items():
            name = display_names.get(key, key)
            
            if data['sample_incorrect']:
                print(f"\n[{name}] Incorrect Predictions:")
                for sample in data['sample_incorrect'][:3]:
                    print(f"  {sample['input']} → Predicted: {sample['predicted']}, "
                          f"Expected: {sample['expected']}")
            
            if data['sample_correct'] and not data['sample_incorrect']:
                print(f"\n[{name}] Correct Predictions (sample):")
                for sample in data['sample_correct'][:3]:
                    print(f"  {sample['input']} → {sample['predicted']} ✓")
    
    def interactive_test(self):
        """
        Run interactive testing mode for manual problem input.
        """
        print("\n" + "=" * 50)
        print("Interactive Addition Testing")
        print("=" * 50)
        print("Enter problems like '123+456=' or 'quit' to exit\n")
        
        while True:
            try:
                user_input = input("Problem: ").strip()
            except EOFError:
                break
            
            # Check for exit commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            # Skip empty input
            if not user_input:
                continue
            
            # Validate format
            if not user_input.endswith("="):
                print("  Invalid format. Use 'NUM1+NUM2=' (e.g., '123+456=')")
                continue
            
            # Parse and validate the problem
            try:
                # Remove the trailing '='
                problem = user_input[:-1]
                
                # Split by '+'
                if '+' not in problem:
                    print("  Invalid format. Must include '+' operator.")
                    continue
                
                parts = problem.split('+')
                if len(parts) != 2:
                    print("  Invalid format. Use 'NUM1+NUM2=' (e.g., '123+456=')")
                    continue
                
                num1_str, num2_str = parts[0].strip(), parts[1].strip()
                
                # Verify both are valid numbers
                if not num1_str.isdigit() or not num2_str.isdigit():
                    print("  Invalid numbers. Both operands must be positive integers.")
                    continue
                
                num1 = int(num1_str)
                num2 = int(num2_str)
                
            except ValueError:
                print("  Invalid input. Please try again.")
                continue
            
            # Generate prediction
            predicted = self.predict(user_input)
            
            # Calculate actual answer
            actual = str(num1 + num2)
            
            # Display results
            is_correct = predicted == actual
            print(f"  Predicted: {predicted}")
            print(f"  Actual:    {actual}")
            print(f"  {'✓ Correct' if is_correct else '✗ Incorrect'}\n")


def main():
    """Main entry point for the evaluation script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate trained addition transformer"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu), auto-detect if not specified"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive testing mode"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation_results.json",
        help="Output path for results"
    )
    
    args = parser.parse_args()
    
    # Create evaluator
    try:
        evaluator = Evaluator(args.model_path, args.device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease train a model first using: python train.py")
        return
    
    # Run evaluation
    if args.interactive:
        evaluator.interactive_test()
    else:
        try:
            results = evaluator.run_full_evaluation()
            evaluator.print_results_table(results)
            evaluator.save_results(results, args.output)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return


if __name__ == "__main__":
    main()

