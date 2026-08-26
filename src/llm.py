"""Local causal-LM chat wrapper for grounded generation.

Loads a small open-source instruct model (fp16, or 4-bit NF4 via bitsandbytes for
larger models such as Phi-3-mini) and exposes deterministic greedy generation plus
temperature sampling used by the semantic-entropy abstention guard. Captures
latency and token-throughput so the two-model comparison can report cost as
tokens/sec + VRAM rather than dollars. Adapted from my llm-uncertainty-quantification project.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import (
    GENERATION_MODEL_A,
    MAX_NEW_TOKENS,
    NUM_ABSTENTION_SAMPLES,
    SAMPLE_TEMPERATURE,
    SAMPLE_TOP_P,
    SEED,
)


@dataclass(frozen=True)
class GenerationResult:
    """A generated answer with timing and token-usage statistics."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float

    @property
    def tokens_per_second(self) -> float:
        """Completion throughput; 0.0 when no tokens or time were measured."""
        if self.latency_seconds <= 0.0:
            return 0.0
        return self.completion_tokens / self.latency_seconds


class LocalLLM:
    """Small instrumented instruct model for grounded RAG generation."""

    def __init__(
        self,
        model_id: str = GENERATION_MODEL_A,
        device: str = "cuda",
        load_in_4bit: bool = False,
    ) -> None:
        """Load tokenizer + model (fp16, or 4-bit NF4 when load_in_4bit)."""
        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if load_in_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=quant, device_map={"": 0}
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)
            self.model.to(device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _encode(self, messages: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        """Apply the chat template to a message list and tokenize to tensors."""
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer(text, return_tensors="pt").to(self.model.device)

    @torch.no_grad()
    def generate(
        self, messages: list[dict[str, str]], max_new_tokens: int = MAX_NEW_TOKENS
    ) -> GenerationResult:
        """Greedy-decode a chat message list; returns text + timing/token stats."""
        inputs = self._encode(messages)
        prompt_len = int(inputs.input_ids.shape[1])
        start = time.perf_counter()
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        latency = time.perf_counter() - start
        gen_ids = out[0][prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_len,
            completion_tokens=int(gen_ids.shape[0]),
            latency_seconds=latency,
        )

    @torch.no_grad()
    def sample(
        self, messages: list[dict[str, str]], num_samples: int = NUM_ABSTENTION_SAMPLES
    ) -> list[str]:
        """Draw stochastic samples of the same prompt for semantic-entropy checks."""
        inputs = self._encode(messages)
        prompt_len = int(inputs.input_ids.shape[1])
        torch.manual_seed(SEED)
        out = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=SAMPLE_TEMPERATURE,
            top_p=SAMPLE_TOP_P,
            num_return_sequences=num_samples,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return [
            self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
            for seq in out
        ]

    def count_tokens(self, text: str) -> int:
        """Return the tokenizer's token count for a string (for budgeting)."""
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)
