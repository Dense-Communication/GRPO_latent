"""
RL-enabled LatentMAS Method with Reading Policy.

Extends the base LatentMASMethod to support:
- Semantic memory block segmentation
- Selective block reading via learned policy
- Trajectory collection for RL training
"""

from typing import Dict, List, Optional, Tuple

import torch
import argparse

from . import default_agents
from .latent_mas import LatentMASMethod
from models import ModelWrapper, _past_length
from prompts import build_agent_message_sequential_latent_mas, build_agent_message_hierarchical_latent_mas
from utils import extract_gsm8k_answer, normalize_answer, extract_markdown_python_block, run_with_timeout, extract_choice_answer, extract_winogrande_answer

from memory import MemoryPool, SemanticBlockSegmenter
from policy import ReadingPolicyNetwork, BlockSelector
from training import Transition

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None


class LatentMASMethodRL(LatentMASMethod):
    """
    LatentMAS with RL-based reading policy.

    Key differences from base LatentMASMethod:
    1. Segments KV/embeddings into semantic blocks after each agent
    2. Uses reading policy to select top-k blocks for judger
    3. Records trajectories for RL training

    Supports both HF and vLLM backends.
    """

    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 10,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
        # RL-specific arguments
        reading_policy: Optional[ReadingPolicyNetwork] = None,
        segmenter: Optional[SemanticBlockSegmenter] = None,
        block_selector: Optional[BlockSelector] = None,
        top_k_blocks: int = 4,
        rl_training: bool = False,
    ) -> None:
        """
        Args:
            model: Model wrapper
            latent_steps: Number of latent steps
            judger_max_new_tokens: Max tokens for judger generation
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            generate_bs: Batch size
            args: Command line arguments
            reading_policy: Optional trained reading policy network
            segmenter: Optional semantic block segmenter
            block_selector: Optional block selector
            top_k_blocks: Number of blocks to select
            rl_training: Whether in RL training mode (enables trajectory recording)
        """
        super().__init__(
            model,
            latent_steps=latent_steps,
            judger_max_new_tokens=judger_max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            generate_bs=generate_bs,
            args=args,
        )

        # RL components
        self.reading_policy = reading_policy
        self.segmenter = segmenter or SemanticBlockSegmenter(
            similarity_threshold=getattr(args, 'similarity_threshold', 0.85),
            min_block_size=getattr(args, 'min_block_size', 4),
            max_block_size=getattr(args, 'max_block_size', 64),
            layer_idx=getattr(args, 'segment_layer_idx', 16),
        )
        self.block_selector = block_selector or BlockSelector(default_k=top_k_blocks)
        self.top_k_blocks = top_k_blocks
        self.rl_training = rl_training

        # Memory pool for current batch
        self.memory_pool: Optional[MemoryPool] = None

        # Trajectory info (recorded during rl_training mode)
        self._trajectory_info: Optional[Dict] = None

        # Last run statistics (for efficiency tracking in eval mode)
        self._last_run_stats: Dict = {"total_blocks": 0, "selected_blocks": 0}

        # Device
        self.policy_device = args.device if args else 'cuda'

    def _init_memory_pool(self, batch_size: int) -> None:
        """Initialize a fresh memory pool for the batch."""
        self.memory_pool = MemoryPool(batch_size=batch_size, device=self.policy_device)
        self.segmenter.reset_counter()

    def _get_query_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get query embedding from input IDs for policy decision."""
        with torch.no_grad():
            if hasattr(self.model, 'HF_model'):
                embeds = self.model.embedding_layer(input_ids)
            else:
                embeds = self.model.model.get_input_embeddings()(input_ids)
            # Mean pool over sequence length
            return embeds.mean(dim=1)  # [B, D]

    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        """
        Run batch with reading policy (HF mode).

        Overrides base class to add:
        1. Memory pool management
        2. Block-based reading for judger
        3. Trajectory recording
        """
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        # Initialize memory pool
        self._init_memory_pool(batch_size)

        for agent in self.agents:
            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(
                        role=agent.role,
                        question=item["question"],
                        context="",
                        method=self.method_name,
                        args=self.args
                    )
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(
                        role=agent.role,
                        question=item["question"],
                        context="",
                        method=self.method_name,
                        args=self.args
                    )
                    for item in items
                ]

            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if agent.role != "judger":
                # Non-judger: generate latent and add to memory pool
                prev_past_len = _past_length(past_kv)

                if self.args.think:
                    wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                else:
                    wrapped_prompts = prompts

                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.device)

                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                # Generate latent
                past_kv = self.model.generate_latent_batch(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=self.latent_steps,
                    past_key_values=past_kv,
                )

                # Add to memory pool (segment into blocks)
                self.memory_pool.add_agent_memory(
                    past_kv,
                    agent.name,
                    self.segmenter
                )

                # Handle truncation modes
                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                    past_kv = self._truncate_past(past_kv, tokens_to_keep)

                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append({
                        "name": agent.name,
                        "role": agent.role,
                        "input": wrapped_prompts[idx],
                        "input_ids": trimmed_ids,
                        "input_tokens": wrapped_tokens_batch[idx],
                        "latent_steps": self.latent_steps,
                        "output": "",
                    })

            else:
                # Judger: use reading policy to select blocks
                if self.args.think:
                    judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                else:
                    judger_prompts = prompts

                judger_encoded = self.model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                judger_ids = judger_encoded["input_ids"].to(self.model.device)
                judger_mask = judger_encoded["attention_mask"].to(self.model.device)

                judger_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(judger_ids, judger_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    judger_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                # Determine which KV to use for generation
                if self.reading_policy is not None and self.memory_pool.num_blocks > 0:
                    # Use reading policy to select blocks
                    past_for_decoding = self._select_and_get_kv(
                        judger_ids,
                        batch_size,
                    )
                else:
                    # Use full KV (fallback)
                    past_for_decoding = past_kv if self.latent_steps > 0 else None

                # Generate
                generated_batch, _ = self.model.generate_text_batch(
                    judger_ids,
                    judger_mask,
                    max_new_tokens=self.judger_max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    past_key_values=past_for_decoding,
                )

                for idx in range(batch_size):
                    final_text = generated_batch[idx].strip()
                    final_texts[idx] = final_text
                    mask = judger_mask[idx].bool()
                    trimmed_ids = judger_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append({
                        "name": agent.name,
                        "role": agent.role,
                        "input": judger_prompts[idx],
                        "input_ids": trimmed_ids,
                        "input_tokens": judger_tokens_batch[idx],
                        "output": final_text,
                    })

        # Build results
        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]

            if self.task in ['mbppplus', 'humanevalplus']:
                pred = extract_markdown_python_block(final_text)
                gold = item.get("gold", "")
                if pred is None:
                    ok = False
                    error_msg = "python error: No python code block found"
                else:
                    python_code_to_exe = pred + "\n" + gold
                    ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)
                print(f'=========================================')
                print(f'Question {idx}')
                print(f'error_msg: {error_msg}')

            elif self.task in ["aime2024", "aime2025"]:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = str(item.get("gold", "")).strip()
                try:
                    pred_int = int(pred)
                    gold_int = int(gold)
                    ok = (pred_int == gold_int)
                    error_msg = None
                except ValueError:
                    ok = False
                    error_msg = f'Value error in parsing answer. Pred: {pred}, Gold: {gold}'

            elif self.task in ["gpqa", "arc_easy", "arc_challenge", "medqa"]:
                # Multiple choice tasks - extract A/B/C/D
                pred = extract_choice_answer(final_text)
                pred = normalize_answer(pred) if pred else None
                gold = normalize_answer(item.get("gold", ""))
                ok = (pred == gold) if (pred and gold) else False
                error_msg = None

            elif self.task == "winogrande":
                # Winogrande uses 1/2 as answers
                pred = extract_winogrande_answer(final_text)
                pred = normalize_answer(pred) if pred else None
                gold = normalize_answer(item.get("gold", ""))
                ok = (pred == gold) if (pred and gold) else False
                error_msg = None

            else:
                # Default: GSM8K style numeric answer
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False
                error_msg = None

            results.append({
                "question": item["question"],
                "gold": gold,
                "solution": item["solution"],
                "prediction": pred,
                "raw_prediction": final_text,
                "agents": agent_traces[idx],
                "correct": ok,
            })

        return results

    def _select_and_get_kv(
        self,
        judger_ids: torch.Tensor,
        batch_size: int,
    ) -> Optional[Tuple]:
        """
        Use reading policy to select blocks and return selected KV.

        Also records trajectory info if rl_training is enabled.
        """
        # Get query embedding
        query_embed = self._get_query_embedding(judger_ids)  # [B, D]

        # Get block summaries
        block_summaries = self.memory_pool.get_block_summaries()  # [B, num_blocks, D]

        if block_summaries.size(1) == 0:
            return None

        # Move to policy device
        query_embed = query_embed.to(self.policy_device)
        block_summaries = block_summaries.to(self.policy_device)

        # Get policy output
        self.reading_policy.eval()
        logits, probs = self.reading_policy(query_embed, block_summaries)

        # Select top-k blocks
        k = min(self.top_k_blocks, block_summaries.size(1))
        selected_indices, log_probs = self.block_selector.select_top_k(
            probs,
            k=k,
            training=self.rl_training,
            return_log_probs=True,
        )

        # Record trajectory info if training
        if self.rl_training:
            self._trajectory_info = {
                "query_embed": query_embed.detach().cpu(),
                "block_summaries": block_summaries.detach().cpu(),
                "selected_indices": selected_indices.detach().cpu(),
                "log_probs": log_probs.detach().cpu(),
                "probs": probs.detach().cpu(),
                "num_blocks": block_summaries.size(1),
            }

        # Get selected KV
        selected_kv = self.memory_pool.get_selected_kv(selected_indices)

        return selected_kv

    def run_batch_vllm(self, items: List[Dict]) -> List[Dict]:
        """
        Run batch with reading policy (vLLM mode).

        Uses embedding-based memory instead of KV cache.
        """
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        # Initialize memory pool
        self._init_memory_pool(batch_size)

        embedding_record = []

        for agent in self.agents:
            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(
                        role=agent.role,
                        question=item["question"],
                        context="",
                        method=self.method_name,
                        args=self.args
                    )
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(
                        role=agent.role,
                        question=item["question"],
                        context="",
                        method=self.method_name,
                        args=self.args
                    )
                    for item in items
                ]

            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if agent.role != "judger":
                prev_past_len = _past_length(past_kv)

                if self.args.think:
                    wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                else:
                    wrapped_prompts = prompts

                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.HF_device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.HF_device)

                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                # Generate latent with hidden state
                past_kv, previous_hidden_embedding = self.model.generate_latent_batch_hidden_state(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=self.latent_steps,
                    past_key_values=past_kv,
                )

                # Add embeddings to memory pool
                self.memory_pool.add_agent_embedding(
                    previous_hidden_embedding,
                    agent.name,
                    self.segmenter,
                )

                # Handle truncation
                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                    past_kv = self._truncate_past(past_kv, tokens_to_keep)

                if self.latent_only:
                    if self.latent_steps > 0:
                        previous_hidden_embedding = previous_hidden_embedding[:, -self.latent_steps:, :]
                    else:
                        previous_hidden_embedding = previous_hidden_embedding[:, 0:0, :]

                embedding_record.append(previous_hidden_embedding)

                if self.sequential_info_only or self.latent_only:
                    embedding_record = embedding_record[-1:]

                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append({
                        "name": agent.name,
                        "role": agent.role,
                        "input": wrapped_prompts[idx],
                        "input_ids": trimmed_ids,
                        "input_tokens": wrapped_tokens_batch[idx],
                        "latent_steps": self.latent_steps,
                        "output": "",
                    })

            else:
                # Judger with reading policy
                if self.args.think:
                    judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                else:
                    judger_prompts = prompts

                judger_encoded = self.model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                judger_ids = judger_encoded["input_ids"].to(self.model.HF_device)

                # Get current prompt embedding
                curr_prompt_emb = self.model.embedding_layer(judger_ids).to(self.vllm_device)

                # Select embeddings using reading policy
                if self.reading_policy is not None and self.memory_pool.num_blocks > 0:
                    selected_embeddings = self._select_embeddings_for_vllm(
                        judger_ids, batch_size
                    )
                elif self.memory_pool.num_blocks > 0:
                    # Baseline mode: read ALL blocks (full memory access)
                    # This serves as the upper bound for accuracy
                    # Policy should achieve similar accuracy with fewer blocks
                    num_blocks = self.memory_pool.num_blocks
                    # Select ALL blocks for baseline
                    all_indices = torch.arange(num_blocks, device=self.vllm_device)
                    all_indices = all_indices.unsqueeze(0).expand(batch_size, -1)
                    selected_embeddings = self.memory_pool.get_selected_embeddings(all_indices)
                    selected_embeddings = selected_embeddings.to(self.vllm_device)
                    # Record stats for efficiency tracking
                    self._last_run_stats = {"total_blocks": num_blocks, "selected_blocks": num_blocks}
                else:
                    # Fallback: use embedding_record directly
                    selected_embeddings = torch.cat(embedding_record, dim=1).to(self.vllm_device)

                # Insert latent embeddings into prompt
                whole_prompt_emb = self._insert_latent_embeddings(
                    curr_prompt_emb,
                    selected_embeddings,
                    judger_prompts,
                )

                # vLLM generation
                prompt_embeds_list = [
                    {"prompt_embeds": embeds}
                    for embeds in whole_prompt_emb
                ]

                outputs = self.model.vllm_engine.generate(
                    prompt_embeds_list,
                    self.sampling_params,
                )

                generated_texts = [out.outputs[0].text.strip() for out in outputs]

                for idx in range(batch_size):
                    text_out = generated_texts[idx].strip()
                    final_texts[idx] = text_out
                    agent_traces[idx].append({
                        "name": agent.name,
                        "role": agent.role,
                        "input": judger_prompts[idx],
                        "output": text_out,
                    })

        # Build results
        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            gold = item["gold"]

            if self.task in ["gpqa", "arc_easy", "arc_challenge", "medqa"]:
                # Multiple choice tasks
                pred = extract_choice_answer(final_text)
                pred = normalize_answer(pred) if pred else None
                gold = normalize_answer(gold)
                ok = (pred == gold) if (pred and gold) else False
            elif self.task == "winogrande":
                # Winogrande uses 1/2 as answers
                pred = extract_winogrande_answer(final_text)
                pred = normalize_answer(pred) if pred else None
                gold = normalize_answer(gold)
                ok = (pred == gold) if (pred and gold) else False
            elif self.task in ["aime2024", "aime2025"]:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                try:
                    ok = (int(pred) == int(gold)) if pred else False
                except ValueError:
                    ok = False
            else:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                ok = (pred == gold) if (pred and gold) else False

            results.append({
                "question": item["question"],
                "gold": gold,
                "solution": item["solution"],
                "prediction": pred,
                "raw_prediction": final_text,
                "agents": agent_traces[idx],
                "correct": ok,
            })

        return results

    def _select_embeddings_for_vllm(
        self,
        judger_ids: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Use reading policy to select embeddings for vLLM mode.
        """
        # Get query embedding
        query_embed = self._get_query_embedding(judger_ids)  # [B, D]

        # Get block summaries
        block_summaries = self.memory_pool.get_block_summaries()  # [B, num_blocks, D]

        if block_summaries.size(1) == 0:
            return torch.zeros(batch_size, 0, query_embed.size(-1), device=query_embed.device)

        # Move to policy device
        query_embed = query_embed.to(self.policy_device)
        block_summaries = block_summaries.to(self.policy_device)

        # Get policy output
        self.reading_policy.eval()
        logits, probs = self.reading_policy(query_embed, block_summaries)

        # Select top-k blocks
        k = min(self.top_k_blocks, block_summaries.size(1))
        selected_indices, log_probs = self.block_selector.select_top_k(
            probs,
            k=k,
            training=self.rl_training,
            return_log_probs=True,
        )

        # Record trajectory info if training
        if self.rl_training:
            self._trajectory_info = {
                "query_embed": query_embed.detach().cpu(),
                "block_summaries": block_summaries.detach().cpu(),
                "selected_indices": selected_indices.detach().cpu(),
                "log_probs": log_probs.detach().cpu(),
                "probs": probs.detach().cpu(),
                "num_blocks": block_summaries.size(1),
            }

        # Record stats for efficiency tracking (always)
        self._last_run_stats = {"total_blocks": block_summaries.size(1), "selected_blocks": k}

        # Get selected embeddings and move to vLLM device
        selected_embeddings = self.memory_pool.get_selected_embeddings(selected_indices)
        selected_embeddings = selected_embeddings.to(self.vllm_device)

        return selected_embeddings

    def _insert_latent_embeddings(
        self,
        curr_prompt_emb: torch.Tensor,
        latent_embeddings: torch.Tensor,
        judger_prompts: List[str],
    ) -> torch.Tensor:
        """
        Insert latent embeddings into prompt at the appropriate position.

        For Qwen models, inserts after "<|im_start|>user\n".
        """
        assert "Qwen" in self.args.model_name or "qwen" in self.args.model_name, \
            "latent_embedding_position is only supported for Qwen models currently."

        # Ensure latent_embeddings is on the same device as curr_prompt_emb
        if latent_embeddings.device != curr_prompt_emb.device:
            latent_embeddings = latent_embeddings.to(curr_prompt_emb.device)

        B, L, H = curr_prompt_emb.shape
        _, Lp, _ = latent_embeddings.shape

        # Find insertion positions
        len_of_left = []
        for p in judger_prompts:
            idx = p.find("<|im_start|>user\n")
            left = p[: idx + len("<|im_start|>user\n")]
            len_of_left.append(len(self.model.tokenizer(left)['input_ids']))

        # Build combined embeddings
        whole_prompt_emb_list = []
        for i in range(B):
            insert_idx = len_of_left[i]
            left_emb = curr_prompt_emb[i, :insert_idx, :]
            right_emb = curr_prompt_emb[i, insert_idx:, :]
            combined = torch.cat([left_emb, latent_embeddings[i], right_emb], dim=0)
            whole_prompt_emb_list.append(combined)

        # Pad to max length
        max_len = max(x.shape[0] for x in whole_prompt_emb_list)
        whole_prompt_emb = torch.stack([
            torch.cat([x, torch.zeros(max_len - x.shape[0], H, device=x.device)], dim=0)
            for x in whole_prompt_emb_list
        ])

        return whole_prompt_emb

    def get_trajectory_info(self) -> Optional[Dict]:
        """Get trajectory info from last run (for RL training)."""
        return self._trajectory_info

    def get_efficiency_stats(self) -> Dict:
        """
        Get efficiency statistics for comparison experiments.

        Returns:
            Dict with:
            - total_blocks: Total number of memory blocks available
            - selected_blocks: Number of blocks selected by policy
            - selection_ratio: selected_blocks / total_blocks
            - latent_steps: Number of latent steps per agent
            - total_latent_tokens: Approximate total latent tokens used
        """
        total = self._last_run_stats.get("total_blocks", 0)
        selected = self._last_run_stats.get("selected_blocks", 0)

        stats = {
            "total_blocks": total,
            "selected_blocks": selected,
            "selection_ratio": selected / max(total, 1),
            "latent_steps": self.latent_steps,
            "top_k_blocks": self.top_k_blocks,
            "has_reading_policy": self.reading_policy is not None,
        }

        return stats

    def create_transition(
        self,
        reward: float,
        task_reward: float = 0.0,
        consistency_reward: float = 0.0,
        cost_penalty: float = 0.0,
    ) -> Optional[Transition]:
        """
        Create a Transition from the last run's trajectory info.

        Args:
            reward: Total reward
            task_reward: Task correctness component
            consistency_reward: Evidence consistency component
            cost_penalty: Read cost component

        Returns:
            Transition object or None if no trajectory info available
        """
        if self._trajectory_info is None:
            return None

        return Transition(
            query_embed=self._trajectory_info["query_embed"][0],  # First batch item
            block_summaries=self._trajectory_info["block_summaries"][0],
            selected_indices=self._trajectory_info["selected_indices"][0],
            log_probs=self._trajectory_info["log_probs"][0],
            probs=self._trajectory_info["probs"][0] if "probs" in self._trajectory_info else None,
            reward=reward,
            task_reward=task_reward,
            consistency_reward=consistency_reward,
            cost_penalty=cost_penalty,
            metadata={
                "num_blocks": self._trajectory_info["num_blocks"],
            }
        )
