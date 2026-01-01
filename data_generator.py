"""
Data generator for synthetic addition datasets.

This module creates training, validation, and test datasets for the
3-digit addition task. It supports various sampling strategies and
can generate specialized test sets for evaluating model generalization.
"""

import random
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any

from config import TaskConfig, DataConfig


class AdditionDataGenerator:
    """
    Generator for synthetic addition datasets.
    
    Creates pairs of k-digit numbers and their sums, with support for
    various sampling strategies (uniform, carries, edge cases).
    """
    
    def __init__(self, k_digits: int = None, seed: int = None):
        """
        Initialize the data generator.
        
        Args:
            k_digits: Number of digits per operand (default: TaskConfig.K_DIGITS)
            seed: Random seed for reproducibility (default: DataConfig.SEED)
        """
        # Use defaults from config if not provided
        self.k_digits = k_digits if k_digits is not None else TaskConfig.K_DIGITS
        seed = seed if seed is not None else DataConfig.SEED
        
        # Calculate value range for k-digit numbers
        # e.g., for k=3: min_val=100, max_val=999
        self.min_val = 10 ** (self.k_digits - 1)
        self.max_val = 10 ** self.k_digits - 1
        
        # Set random seeds for reproducibility
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass  # numpy not required
    
    def generate_pair(self, strategy: str = "uniform") -> Tuple[int, int]:
        """
        Generate a pair of k-digit numbers using the specified strategy.
        
        Args:
            strategy: Sampling strategy - "uniform", "carries", or "edge_cases"
            
        Returns:
            Tuple of two k-digit integers (a, b)
        """
        if strategy == "uniform":
            # Uniform random sampling from the full range
            a = random.randint(self.min_val, self.max_val)
            b = random.randint(self.min_val, self.max_val)
            
        elif strategy == "carries":
            # Generate numbers where at least one digit pair sums to >= 10
            # This ensures the addition requires carrying
            while True:
                a = random.randint(self.min_val, self.max_val)
                b = random.randint(self.min_val, self.max_val)
                if self.requires_carry(a, b):
                    break
                    
        elif strategy == "edge_cases":
            # Generate numbers with many 9s or 0s (boundary cases)
            edge_type = random.choice(["nines", "zeros", "mixed"])
            
            if edge_type == "nines":
                # Numbers with multiple 9s (likely to cause carries)
                a = self._generate_with_digit(9, min_count=2)
                b = self._generate_with_digit(9, min_count=1)
            elif edge_type == "zeros":
                # Numbers with trailing/leading zeros patterns
                a = self._generate_with_digit(0, min_count=1, allow_leading=False)
                b = self._generate_with_digit(0, min_count=1, allow_leading=False)
            else:
                # Mix of 9s and 0s
                a = self._generate_edge_mixed()
                b = self._generate_edge_mixed()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return (a, b)
    
    def _generate_with_digit(
        self, 
        digit: int, 
        min_count: int = 1, 
        allow_leading: bool = True
    ) -> int:
        """
        Generate a number containing at least min_count of the specified digit.
        
        Args:
            digit: The digit to include (0-9)
            min_count: Minimum occurrences of the digit
            allow_leading: Whether the digit can be in the leading position
            
        Returns:
            A k-digit integer with the specified digit pattern
        """
        digits = []
        positions = list(range(self.k_digits))
        
        # If digit is 0 and can't be leading, exclude position 0
        if digit == 0 and not allow_leading:
            available_positions = positions[1:]
        else:
            available_positions = positions
        
        # Select positions for the target digit
        if len(available_positions) >= min_count:
            target_positions = set(random.sample(available_positions, min_count))
        else:
            target_positions = set(available_positions)
        
        # Build the number digit by digit
        for pos in range(self.k_digits):
            if pos in target_positions:
                digits.append(digit)
            elif pos == 0:
                # First digit can't be 0
                digits.append(random.randint(1, 9))
            else:
                digits.append(random.randint(0, 9))
        
        return int("".join(map(str, digits)))
    
    def _generate_edge_mixed(self) -> int:
        """Generate a number with a mix of 9s, 0s, and random digits."""
        digits = []
        for pos in range(self.k_digits):
            if pos == 0:
                # First digit: either 9 or random 1-9
                digits.append(random.choice([9, random.randint(1, 9)]))
            else:
                # Other digits: 0, 9, or random
                choice = random.choice(["zero", "nine", "random"])
                if choice == "zero":
                    digits.append(0)
                elif choice == "nine":
                    digits.append(9)
                else:
                    digits.append(random.randint(0, 9))
        
        return int("".join(map(str, digits)))
    
    def format_example(self, a: int, b: int) -> Tuple[str, str]:
        """
        Format a number pair as input/output strings.
        
        Args:
            a: First operand
            b: Second operand
            
        Returns:
            Tuple of (input_str, output_str)
            - input_str: "123+456=" format (no spaces)
            - output_str: "579" (the sum)
        """
        # Format without spaces (TaskConfig.USE_SPACES would be False)
        input_str = f"{a}+{b}="
        output_str = str(a + b)
        
        return (input_str, output_str)
    
    def requires_carry(self, a: int, b: int) -> bool:
        """
        Check if adding a and b requires at least one carry operation.
        
        Args:
            a: First operand
            b: Second operand
            
        Returns:
            True if any digit pair sums to >= 10
        """
        # Convert to strings and pad to same length
        str_a = str(a)
        str_b = str(b)
        max_len = max(len(str_a), len(str_b))
        str_a = str_a.zfill(max_len)
        str_b = str_b.zfill(max_len)
        
        # Check each digit position from right to left
        carry = 0
        for i in range(max_len - 1, -1, -1):
            digit_a = int(str_a[i])
            digit_b = int(str_b[i])
            digit_sum = digit_a + digit_b + carry
            
            if digit_sum >= 10:
                return True
            carry = 0  # No carry happened at this position
        
        return False
    
    def generate_dataset(
        self, 
        size: int, 
        strategy: str = "uniform", 
        allow_duplicates: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate a dataset of addition examples.
        
        Args:
            size: Number of examples to generate
            strategy: Sampling strategy ("uniform", "carries", "edge_cases")
            allow_duplicates: Whether to allow duplicate number pairs
            
        Returns:
            List of dicts with keys: input, output, a, b, result, requires_carry
        """
        dataset = []
        seen_pairs = set()  # Track (min(a,b), max(a,b)) to catch both orderings
        
        attempts = 0
        max_attempts = size * 100  # Prevent infinite loops
        
        while len(dataset) < size and attempts < max_attempts:
            attempts += 1
            
            # Generate a pair
            a, b = self.generate_pair(strategy)
            
            # Check for duplicates if not allowed
            if not allow_duplicates:
                # Normalize pair order for duplicate detection
                pair_key = (min(a, b), max(a, b))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
            
            # Format the example
            input_str, output_str = self.format_example(a, b)
            
            # Check if carry is required
            carry_required = self.requires_carry(a, b)
            
            # Create the example dict
            example = {
                "input": input_str,
                "output": output_str,
                "a": a,
                "b": b,
                "result": a + b,
                "requires_carry": carry_required
            }
            
            dataset.append(example)
        
        if len(dataset) < size:
            print(f"Warning: Could only generate {len(dataset)} unique examples "
                  f"(requested {size})")
        
        return dataset
    
    def generate_length_generalization_test(
        self, 
        k_test: int, 
        size: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Generate a test set with different number of digits for length generalization.
        
        Args:
            k_test: Number of digits for the test set
            size: Number of examples to generate
            
        Returns:
            List of examples with k_test-digit numbers
        """
        # Save original values
        original_k = self.k_digits
        original_min = self.min_val
        original_max = self.max_val
        
        # Update for new digit count
        self.k_digits = k_test
        self.min_val = 10 ** (k_test - 1)
        self.max_val = 10 ** k_test - 1
        
        # Generate dataset
        dataset = self.generate_dataset(size, "uniform", allow_duplicates=False)
        
        # Restore original values
        self.k_digits = original_k
        self.min_val = original_min
        self.max_val = original_max
        
        return dataset
    
    def generate_distribution_shift_test(
        self, 
        size: int = 1000, 
        shift_type: str = "many_nines"
    ) -> List[Dict[str, Any]]:
        """
        Generate a test set with distribution shift for robustness testing.
        
        Args:
            size: Number of examples to generate
            shift_type: Type of distribution shift - "many_nines" or "small_numbers"
            
        Returns:
            List of examples with shifted distribution
        """
        dataset = []
        seen_pairs = set()
        
        attempts = 0
        max_attempts = size * 100
        
        while len(dataset) < size and attempts < max_attempts:
            attempts += 1
            
            if shift_type == "many_nines":
                # Generate numbers where at least 2 digits are 9
                a = self._generate_with_digit(9, min_count=2)
                b = self._generate_with_digit(9, min_count=2)
                
            elif shift_type == "small_numbers":
                # Sample from the lower end of the range
                # For k=3: sample from [100, 200]
                range_size = min(100, self.max_val - self.min_val)
                a = random.randint(self.min_val, self.min_val + range_size)
                b = random.randint(self.min_val, self.min_val + range_size)
                
            else:
                raise ValueError(f"Unknown shift_type: {shift_type}")
            
            # Check for duplicates
            pair_key = (min(a, b), max(a, b))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            
            # Format and create example
            input_str, output_str = self.format_example(a, b)
            carry_required = self.requires_carry(a, b)
            
            example = {
                "input": input_str,
                "output": output_str,
                "a": a,
                "b": b,
                "result": a + b,
                "requires_carry": carry_required
            }
            
            dataset.append(example)
        
        if len(dataset) < size:
            print(f"Warning: Could only generate {len(dataset)} unique examples "
                  f"(requested {size})")
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]], filepath: str) -> None:
        """
        Save a dataset to a JSON file.
        
        Args:
            dataset: List of example dictionaries
            filepath: Path to save the file
        """
        # Ensure parent directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Write JSON file
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Saved {len(dataset)} examples to {filepath}")
    
    def load_dataset(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load a dataset from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            List of example dictionaries
        """
        with open(filepath, 'r') as f:
            dataset = json.load(f)
        
        print(f"Loaded {len(dataset)} examples from {filepath}")
        return dataset


def calculate_carry_stats(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics about carry operations in a dataset.
    
    Args:
        dataset: List of example dictionaries
        
    Returns:
        Dict with carry statistics
    """
    total = len(dataset)
    with_carry = sum(1 for ex in dataset if ex["requires_carry"])
    without_carry = total - with_carry
    
    return {
        "total": total,
        "with_carry": with_carry,
        "without_carry": without_carry,
        "carry_ratio": with_carry / total if total > 0 else 0
    }


def generate_all_datasets() -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Generate all standard datasets (train, validation, test).
    
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Create generator
    generator = AdditionDataGenerator()
    
    print("=" * 60)
    print("GENERATING ADDITION DATASETS")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  k_digits: {generator.k_digits}")
    print(f"  Number range: [{generator.min_val}, {generator.max_val}]")
    print()
    
    # Generate training data
    print("[Generating training data...]")
    train_data = generator.generate_dataset(
        size=DataConfig.TRAIN_SIZE,
        strategy="uniform",
        allow_duplicates=False
    )
    
    # Generate validation data
    print("[Generating validation data...]")
    val_data = generator.generate_dataset(
        size=DataConfig.VAL_SIZE,
        strategy="uniform",
        allow_duplicates=False
    )
    
    # Generate test data
    print("[Generating test data...]")
    test_data = generator.generate_dataset(
        size=DataConfig.TEST_SIZE,
        strategy="uniform",
        allow_duplicates=False
    )
    
    # Save datasets (use .json extension for all)
    print("\n[Saving datasets...]")
    generator.save_dataset(train_data, DataConfig.TRAIN_FILE)
    generator.save_dataset(val_data, DataConfig.VAL_FILE)
    # Use .json for test file instead of .txt
    test_file = DataConfig.TEST_FILE.replace(".txt", ".json")
    generator.save_dataset(test_data, test_file)
    
    # Print statistics
    print("\n" + "-" * 40)
    print("DATASET STATISTICS")
    print("-" * 40)
    
    for name, data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        stats = calculate_carry_stats(data)
        print(f"\n{name} set:")
        print(f"  Total examples: {stats['total']:,}")
        print(f"  With carry: {stats['with_carry']:,} ({stats['carry_ratio']:.1%})")
        print(f"  Without carry: {stats['without_carry']:,} ({1 - stats['carry_ratio']:.1%})")
    
    print("\n" + "=" * 60)
    
    return train_data, val_data, test_data


if __name__ == "__main__":
    # Generate standard datasets
    train_data, val_data, test_data = generate_all_datasets()
    
    # Create generator for additional test sets
    generator = AdditionDataGenerator()
    k = TaskConfig.K_DIGITS
    
    print("\n[Generating length generalization tests...]")
    
    # Generate k+1 digit test (4-digit numbers)
    test_k4 = generator.generate_length_generalization_test(k_test=k + 1, size=1000)
    generator.save_dataset(test_k4, f"data/test_length_k{k + 1}.json")
    
    # Generate k+2 digit test (5-digit numbers)
    test_k5 = generator.generate_length_generalization_test(k_test=k + 2, size=1000)
    generator.save_dataset(test_k5, f"data/test_length_k{k + 2}.json")
    
    print("\n[Generating distribution shift tests...]")
    
    # Generate many-nines test
    test_nines = generator.generate_distribution_shift_test(size=1000, shift_type="many_nines")
    generator.save_dataset(test_nines, "data/test_shift_many_nines.json")
    
    # Generate small-numbers test
    test_small = generator.generate_distribution_shift_test(size=1000, shift_type="small_numbers")
    generator.save_dataset(test_small, "data/test_shift_small_numbers.json")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("Done! Generated all datasets successfully.")
    print("=" * 60)

