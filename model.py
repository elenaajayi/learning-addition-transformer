"""
Decoder-only Transformer for 3-digit addition.

This module implements a character-level transformer model that learns
to perform addition of 3-digit numbers. The architecture follows the
standard decoder-only transformer design with causal attention masking.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig, TaskConfig



# Tokenizer


class Tokenizer:
    """
    Character-level tokenizer for the addition task.
    
    Uses the vocabulary defined in TaskConfig which includes:
    - Special tokens: PAD, BOS, EOS
    - Digits: 0-9
    - Operators: +, =
    """
    
    def __init__(self, task_config: TaskConfig = None):
        """Initialize tokenizer with task configuration."""
        self.task_config = task_config or TaskConfig()
        
        # Build vocabulary mappings
        self._token_to_idx = self.task_config.get_token_to_idx()
        self._idx_to_token = self.task_config.get_idx_to_token()
        
        # Special token indices
        self._pad_token_id = self._token_to_idx["PAD"]
        self._bos_token_id = self._token_to_idx["BOS"]
        self._eos_token_id = self._token_to_idx["EOS"]
    
    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self.task_config.get_vocab_size()
    
    @property
    def pad_token_id(self) -> int:
        """Return the padding token index."""
        return self._pad_token_id
    
    @property
    def bos_token_id(self) -> int:
        """Return the beginning-of-sequence token index."""
        return self._bos_token_id
    
    @property
    def eos_token_id(self) -> int:
        """Return the end-of-sequence token index."""
        return self._eos_token_id
    
    def encode(self, text: str) -> list[int]:
        """
        Encode a text string into a list of token IDs.
        
        Args:
            text: Input string (e.g., "123+456=579")
            
        Returns:
            List of token IDs
        """
        token_ids = []
        for char in text:
            if char in self._token_to_idx:
                token_ids.append(self._token_to_idx[char])
            else:
                raise ValueError(f"Unknown character: '{char}'")
        return token_ids
    
    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a list of token IDs back to a string.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded string (special tokens are included)
        """
        tokens = []
        for idx in token_ids:
            if idx in self._idx_to_token:
                token = self._idx_to_token[idx]
                tokens.append(token)
            else:
                raise ValueError(f"Unknown token ID: {idx}")
        return "".join(tokens)
    
    def decode_clean(self, token_ids: list[int]) -> str:
        """
        Decode token IDs, removing special tokens.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded string without special tokens
        """
        tokens = []
        for idx in token_ids:
            if idx in self._idx_to_token:
                token = self._idx_to_token[idx]
                # Skip special tokens
                if token not in self.task_config.SPECIAL_TOKENS:
                    tokens.append(token)
        return "".join(tokens)



# Positional Encoding


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as described in "Attention Is All You Need".
    
    Adds position information to token embeddings using fixed sinusoidal patterns:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Dimension of the model embeddings
            max_seq_len: Maximum sequence length to support
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix [max_seq_len, d_model]
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the division term for the sinusoidal functions
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Token embeddings [batch_size, seq_len, d_model]
            
        Returns:
            Embeddings with positional information added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# Transformer Block


class TransformerBlock(nn.Module):
    """
    A single transformer decoder block with causal self-attention.
    
    Components:
    - Multi-head self-attention with causal masking
    - Feed-forward network with GELU activation
    - Layer normalization (pre-norm style)
    - Residual connections
    - Dropout for regularization
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """
        Initialize transformer block.
        
        Args:
            d_model: Dimension of the model
            n_heads: Number of attention heads
            d_ff: Dimension of the feed-forward hidden layer
            dropout: Dropout probability
        """
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Multi-head self-attention components
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization (pre-norm style)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create a causal attention mask.
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
            
        Returns:
            Causal mask [1, 1, seq_len, seq_len] where True means masked
        """
        # Upper triangular matrix (excluding diagonal) = positions to mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool().unsqueeze(0).unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer block.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Pre-norm for attention
        normed = self.norm1(x)
        
        # Compute Q, K, V projections
        q = self.q_proj(normed)
        k = self.k_proj(normed)
        v = self.v_proj(normed)
        
        # Reshape for multi-head attention: [batch, seq, heads, head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # Now: [batch, heads, seq, head_dim]
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        # attn_scores: [batch, heads, seq, seq]
        
        # Apply causal mask (set masked positions to -inf)
        causal_mask = self._create_causal_mask(seq_len, x.device)
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)
        # attn_output: [batch, heads, seq, head_dim]
        
        # Reshape back: [batch, seq, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output projection and residual connection
        attn_output = self.out_proj(attn_output)
        x = x + self.resid_dropout(attn_output)
        
        # Pre-norm for feed-forward and residual connection
        x = x + self.ff(self.norm2(x))
        
        return x


# Addition Transformer Model


class AdditionTransformer(nn.Module):
    """
    Decoder-only transformer for learning 3-digit addition.
    
    Architecture:
    - Token embeddings
    - Sinusoidal positional encoding
    - Stack of transformer blocks with causal attention
    - Final layer norm
    - Output projection to vocabulary
    
    The model is trained to predict the next token in sequences of the form:
    "BOS 1 2 3 + 4 5 6 = 5 7 9 EOS"
    """
    
    def __init__(self, model_config: ModelConfig, task_config: TaskConfig = None):
        """
        Initialize the addition transformer.
        
        Args:
            model_config: Model architecture configuration
            task_config: Task configuration (for tokenizer)
        """
        super().__init__()
        
        self.model_config = model_config
        self.task_config = task_config or TaskConfig()
        self.tokenizer = Tokenizer(self.task_config)
        
        # Token embeddings
        self.token_embedding = nn.Embedding(
            model_config.VOCAB_SIZE,
            model_config.D_MODEL,
            padding_idx=self.tokenizer.pad_token_id
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=model_config.D_MODEL,
            max_seq_len=model_config.MAX_SEQ_LEN,
            dropout=model_config.DROPOUT
        )
        
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=model_config.D_MODEL,
                n_heads=model_config.N_HEADS,
                d_ff=model_config.D_FF,
                dropout=model_config.DROPOUT
            )
            for _ in range(model_config.N_LAYERS)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(model_config.D_MODEL)
        
        # Output projection to vocabulary
        self.output_proj = nn.Linear(model_config.D_MODEL, model_config.VOCAB_SIZE)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights with Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                # Zero out padding embedding
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            targets: Target token IDs for loss calculation [batch_size, seq_len]
                     (optional, typically shifted input_ids)
            
        Returns:
            Tuple of (logits, loss):
            - logits: Output logits [batch_size, seq_len, vocab_size]
            - loss: Cross-entropy loss if targets provided, else None
        """
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Final layer norm
        x = self.final_norm(x)
        
        # Project to vocabulary
        logits = self.output_proj(x)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            # Flatten for cross-entropy: [batch * seq, vocab_size] vs [batch * seq]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.tokenizer.pad_token_id
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 10,
        temperature: float = 1.0,
        greedy: bool = True
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Starting token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (ignored if greedy=True)
            greedy: If True, use greedy decoding; else sample
            
        Returns:
            Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Get predictions for the current sequence
            # Truncate if exceeding max sequence length
            context = generated
            if context.size(1) > self.model_config.MAX_SEQ_LEN:
                context = context[:, -self.model_config.MAX_SEQ_LEN:]
            
            logits, _ = self.forward(context)
            
            # Get logits for the last position
            next_token_logits = logits[:, -1, :]
            
            if greedy:
                # Greedy decoding: take the most likely token
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                # Sampling with temperature
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences have generated EOS
            if (next_token == self.tokenizer.eos_token_id).all():
                break
        
        return generated
    
    def count_parameters(self) -> int:
        """
        Count the total number of trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test Code


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Addition Transformer Model")
    print("=" * 60)
    
    # Initialize configs
    task_config = TaskConfig()
    model_config = ModelConfig(task_config)
    
    # Create tokenizer
    print("\n[Testing Tokenizer]")
    tokenizer = Tokenizer(task_config)
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  PAD token ID: {tokenizer.pad_token_id}")
    print(f"  BOS token ID: {tokenizer.bos_token_id}")
    print(f"  EOS token ID: {tokenizer.eos_token_id}")
    
    # Test encoding/decoding
    test_text = "123+456=579"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"  Original: '{test_text}'")
    print(f"  Encoded: {encoded}")
    print(f"  Decoded: '{decoded}'")
    assert decoded == test_text, "Encoding/decoding mismatch!"
    print("  ✓ Tokenizer test passed!")
    
    # Create model
    print("\n[Testing Model]")
    model = AdditionTransformer(model_config, task_config)
    num_params = model.count_parameters()
    print(f"  Model parameters: {num_params:,}")
    
    # Test forward pass
    print("\n[Testing Forward Pass]")
    batch_size = 4
    seq_len = 16
    
    # Create dummy input
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits, loss = model(input_ids, targets)
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print("  ✓ Forward pass test passed!")
    
    # Test generation
    print("\n[Testing Generation]")
    # Encode a problem (prepend BOS token manually since it's a special token)
    problem = "123+456="
    problem_ids = [tokenizer.bos_token_id] + tokenizer.encode(problem)
    input_ids = torch.tensor([problem_ids])
    print(f"  Input: 'BOS{problem}'")
    print(f"  Input IDs: {input_ids.tolist()}")
    
    # Generate
    generated_ids = model.generate(input_ids, max_new_tokens=5, greedy=True)
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print(f"  Generated IDs: {generated_ids.tolist()}")
    print(f"  Generated text: '{generated_text}'")
    print("  ✓ Generation test passed!")
    
    # Test with different devices if available
    print("\n[Device Information]")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Available device: {device}")
    
    if device == "cuda":
        model = model.to(device)
        input_ids = input_ids.to(device)
        logits, _ = model(input_ids)
        print(f"  ✓ CUDA forward pass successful!")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

