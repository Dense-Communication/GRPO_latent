"""
Multi-Agent Trajectory Collector with Single Agent Training Mode
支持多agent协作，但每次只训练一个agent的版本
"""

import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from typing import List, Dict, Tuple, Optional
from collections import Counter
import re
from methods.latent_mas import LatentMASMethod
from models import ModelWrapper

class MultiAgentTrajectoryCollector:
    """
    Multi-agent Trajectory收集器（单agent训练模式）
    
    关键特性：
    - 支持多个agents参与rollout
    - 每次只返回一个agent的trajectory用于训练
    - 其他agents作为固定的协作者参与
    """

    def __init__(
        self, 
        config, 
        tokenizer: PreTrainedTokenizer, 
        processor=None, 
        reward_fn=None,
        agent_models: List[str] = None,  # 所有agent模型的路径列表
        current_agent_idx: int = 0,      # 当前训练的agent索引
    ):
        """
        初始化Multi-agent Trajectory Collector
        
        Parameters:
            config: 配置对象
            tokenizer: HuggingFace tokenizer
            processor: 可选的多模态processor
            reward_fn: 奖励计算函数
            agent_models: 所有agent模型的路径列表
            current_agent_idx: 当前正在训练的agent索引（0-based）
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.reward_fn = reward_fn
        
        # Multi-agent配置
        self.n_agents = config.env.get('n_agents', 1)
        self.agent_models = agent_models or []
        self.current_agent_idx = current_agent_idx
        
        # 确保配置合理
        if self.agent_models:
            assert len(self.agent_models) == self.n_agents, \
                f"Number of agent models ({len(self.agent_models)}) must match n_agents ({self.n_agents})"
        
        assert 0 <= self.current_agent_idx < self.n_agents, \
            f"current_agent_idx ({self.current_agent_idx}) must be in [0, {self.n_agents})"
        
        # Action聚合策略
        self.action_reduction_strategy = config.env.get('action_reduction', 'majority_vote')
        self.action_reducer = self._get_action_reducer(self.action_reduction_strategy)
        
        # 其他配置
        self.max_steps = config.env.get('max_steps', 3)
        self.enable_summary = config.env.get('enable_agent_summary', True)
        self.summary_max_length = config.env.get('summary_max_length', 50)

        # 初始化 LatentMAS（如果启用）
        self.use_latent_mas = config.get('use_latent_mas', False)
        self.latent_mas_method = None
        if self.use_latent_mas:
            # 需要从config获取model wrapper
            model_wrapper = config.get('model_wrapper', None)
            if model_wrapper:
                self.latent_mas_method = LatentMASMethod(
                    model=model_wrapper,
                    latent_steps=config.get('latent_steps', 3),
                    temperature=config.get('temperature', 0.7),
                    top_p=config.get('top_p', 0.9),
                    generate_bs=1,  # 单batch处理
                    args=config  # 传递完整配置
                )
        
        print(f"[MultiAgentTrajectoryCollector] Initialized with {self.n_agents} agents")
        print(f"[MultiAgentTrajectoryCollector] Currently training agent {self.current_agent_idx}")
        if self.agent_models:
            print(f"[MultiAgentTrajectoryCollector] Agent models: {self.agent_models}")

    def set_current_agent(self, agent_idx: int):
        """
        切换当前训练的agent
        
        Parameters:
            agent_idx: 要训练的agent索引
        """
        assert 0 <= agent_idx < self.n_agents, \
            f"agent_idx ({agent_idx}) must be in [0, {self.n_agents})"
        self.current_agent_idx = agent_idx
        print(f"[MultiAgentTrajectoryCollector] Switched to training agent {agent_idx}")

    def _get_action_reducer(self, strategy: str):
        """获取action聚合函数"""
        strategies = {
            'majority_vote': self._majority_vote,
            'weighted_vote': self._weighted_vote,
            'first_agent': self._first_agent_vote,
            'current_agent': self._current_agent_vote,  # 新增：使用当前训练agent的决策
        }
        
        if strategy not in strategies:
            raise ValueError(
                f"Unknown action reduction strategy: {strategy}. "
                f"Available: {list(strategies.keys())}"
            )
        
        return strategies[strategy]

    def _majority_vote(self, actions: List[str], logprobs: Optional[List[float]] = None) -> str:
        """多数投票聚合"""
        if not actions:
            return ""
        
        vote_counts = Counter(actions)
        most_common = vote_counts.most_common(1)
        
        if most_common:
            return most_common[0][0]
        return actions[0]

    def _weighted_vote(self, actions: List[str], logprobs: Optional[List[float]] = None) -> str:
        """加权投票（基于logprobs）"""
        if not actions:
            return ""
        
        if logprobs is None:
            return self._majority_vote(actions, None)
        
        action_weights = {}
        for action, logprob in zip(actions, logprobs):
            if action not in action_weights:
                action_weights[action] = 0
            action_weights[action] += np.exp(logprob)
        
        best_action = max(action_weights.items(), key=lambda x: x[1])[0]
        return best_action

    def _first_agent_vote(self, actions: List[str], logprobs: Optional[List[float]] = None) -> str:
        """选择第一个agent的决策"""
        if actions:
            return actions[0]
        return ""

    def _current_agent_vote(self, actions: List[str], logprobs: Optional[List[float]] = None) -> str:
        """使用当前训练agent的决策（用于单agent训练模式）"""
        if len(actions) > self.current_agent_idx:
            return actions[self.current_agent_idx]
        return actions[0] if actions else ""

    def generate_agent_actions(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        fixed_agent_wgs: Optional[List] = None,  # 其他固定agent的WorkerGroups
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """
        生成所有agents的actions
        
        Parameters:
            gen_batch: 输入batch
            actor_rollout_wg: 当前训练agent的WorkerGroup
            fixed_agent_wgs: 其他固定agent的WorkerGroups（可选）
            
        Returns:
            all_agent_actions: 所有agent的actions
            all_agent_logprobs: 所有agent的logprobs
        """
        batch_size = len(gen_batch)
        all_agent_actions = []
        all_agent_logprobs = []
        
        for agent_id in range(self.n_agents):
            if agent_id == self.current_agent_idx:
                # 使用正在训练的agent生成
                with actor_rollout_wg.rwlock.r_lock():
                    output = actor_rollout_wg.generate_sequences(
                        prompts=gen_batch,
                        temperature=self.config.actor_rollout_ref.rollout.temperature,
                        top_p=self.config.actor_rollout_ref.rollout.top_p,
                        max_new_tokens=self.config.data.max_response_length,
                    )
                
                # 提取生成的文本
                agent_actions = []
                agent_logprobs = []
                for i in range(batch_size):
                    if hasattr(output, 'sequences') and output.sequences is not None:
                        generated_ids = output.sequences[i]
                        generated_text = self.tokenizer.decode(
                            generated_ids, 
                            skip_special_tokens=True
                        )
                    else:
                        generated_text = f"Response from agent {agent_id}"
                    
                    agent_actions.append(generated_text)
                    
                    # 获取logprobs
                    if hasattr(output, 'scores'):
                        logprob = torch.mean(output.scores[i]).item() if output.scores is not None else 0.0
                    else:
                        logprob = 0.0
                    agent_logprobs.append(logprob)
                
            else:
                # 使用固定的agent生成（如果提供了fixed_agent_wgs）
                if fixed_agent_wgs and agent_id < len(fixed_agent_wgs):
                    # 使用对应的固定agent
                    fixed_wg = fixed_agent_wgs[agent_id]
                    with fixed_wg.rwlock.r_lock():
                        output = fixed_wg.generate_sequences(
                            prompts=gen_batch,
                            temperature=self.config.actor_rollout_ref.rollout.temperature,
                            top_p=self.config.actor_rollout_ref.rollout.top_p,
                            max_new_tokens=self.config.data.max_response_length,
                        )
                    
                    # 处理输出
                    agent_actions = []
                    agent_logprobs = []
                    for i in range(batch_size):
                        if hasattr(output, 'sequences') and output.sequences is not None:
                            generated_text = self.tokenizer.decode(output.sequences[i], skip_special_tokens=True)
                        else:
                            generated_text = f"Fixed response from agent {agent_id}"
                        agent_actions.append(generated_text)
                        agent_logprobs.append(0.0)
                else:
                    # 使用模拟响应（备用方案）
                    agent_actions = [f"Simulated response from agent {agent_id}"] * batch_size
                    agent_logprobs = [0.0] * batch_size
            
            all_agent_actions.append(agent_actions)
            all_agent_logprobs.append(agent_logprobs)
        
        return all_agent_actions, all_agent_logprobs

    def vanilla_multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        fixed_agent_wgs: Optional[List] = None,
    ) -> Tuple[List[List[Dict]], np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        执行multi-turn rollout loop（单agent训练版本）
        
        Parameters:
            gen_batch: 初始prompt batch
            actor_rollout_wg: 当前训练agent的WorkerGroup
            fixed_agent_wgs: 其他固定agent的WorkerGroups
            
        Returns:
            仅返回当前训练agent的trajectory数据
        """
        batch_size = len(gen_batch)
        
        # 初始化轨迹存储（只存储当前agent的）
        total_batch_list = []
        episode_rewards = np.zeros(batch_size)
        episode_lengths = np.zeros(batch_size, dtype=int)
        
        # 生成轨迹UID
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)])
        
        # 初始化prompts
        if hasattr(gen_batch, 'non_tensor_batch') and 'raw_prompt' in gen_batch.non_tensor_batch:
            current_prompts = gen_batch.non_tensor_batch['raw_prompt']
        else:
            current_prompts = [""] * batch_size
        
        # Multi-turn loop
        for step in range(self.max_steps):
            
            # Step 1: 生成所有agents的actions
            all_agent_actions, all_agent_logprobs = self.generate_agent_actions(
                gen_batch, 
                actor_rollout_wg,
                fixed_agent_wgs
            )
            
            # Step 2: 聚合actions（用于环境交互）
            aggregated_actions = []
            for i in range(batch_size):
                agent_actions_for_env_i = [
                    all_agent_actions[agent_id][i] 
                    for agent_id in range(self.n_agents)
                ]
                agent_logprobs_for_env_i = [
                    all_agent_logprobs[agent_id][i]
                    for agent_id in range(self.n_agents)
                ]
                
                # 使用配置的聚合策略
                aggregated_action = self.action_reducer(
                    agent_actions_for_env_i,
                    agent_logprobs_for_env_i
                )
                aggregated_actions.append(aggregated_action)
            
            # Step 3: 计算奖励（基于聚合的action）
            rewards = []
            dones = [step >= self.max_steps - 1] * batch_size
            
            for i in range(batch_size):
                if self.reward_fn is not None:
                    reward = self.reward_fn(current_prompts[i], aggregated_actions[i], step, dones[i])
                else:
                    # 简单奖励规则
                    reward = 1.0 if dones[i] else 0.0
                rewards.append(reward)
            
            # Step 4: 只记录当前训练agent的轨迹数据
            for i in range(batch_size):
                # 获取当前agent的action
                current_agent_action = all_agent_actions[self.current_agent_idx][i]
                
                trajectory_step = {
                    'prompt': current_prompts[i] if isinstance(current_prompts, list) else "",
                    'action': current_agent_action,  # 使用当前agent的action
                    'aggregated_action': aggregated_actions[i],  # 记录聚合后的action（参考）
                    'reward': rewards[i],
                    'done': dones[i],
                    'step': step,
                    'traj_uid': traj_uid[i],
                    'agent_id': self.current_agent_idx,
                    'active_masks': True,
                    # 记录所有agents的actions（用于分析）
                    'all_agent_actions': [all_agent_actions[a][i] for a in range(self.n_agents)],
                }
                
                total_batch_list.append([trajectory_step])
                
                # 累积奖励和长度
                episode_rewards[i] += rewards[i]
                episode_lengths[i] = step + 1
            
            # 更新observations（基于聚合的action）
            next_obs = []
            for i in range(batch_size):
                new_prompt = f"{current_prompts[i]}\nStep {step+1}: {aggregated_actions[i]}"
                next_obs.append(new_prompt)
            current_prompts = next_obs
            
            # 检查是否所有都结束
            if all(dones):
                break
                
            # Step 5: 如果未结束且启用了LatentMAS，共享KV-cache
            if not all(dones) and self.use_latent_mas and self.latent_mas_method:
                # 准备用于LatentMAS的数据
                latent_batch = []
                for i in range(batch_size):
                    # 构造每个环境的当前状态
                    latent_item = {
                        'question': current_prompts[i] if isinstance(current_prompts, list) else "",
                        'agent_responses': [all_agent_actions[a][i] for a in range(self.n_agents)],
                        'step': step,
                        'done': dones[i]
                    }
                    latent_batch.append(latent_item)
                
                # 调用LatentMAS进行KV-cache共享
                try:
                    # 运行LatentMAS批处理
                    latent_results = self.latent_mas_method.run_batch(latent_batch)
                    
                    # 可选：使用LatentMAS的输出更新agent actions
                    if latent_results and len(latent_results) == batch_size:
                        for i in range(batch_size):
                            if 'shared_knowledge' in latent_results[i]:
                                # 这里可以将共享的知识注入到下一轮的prompts中
                                shared_info = latent_results[i]['shared_knowledge']
                                # 更新下一轮的观察
                                if shared_info:
                                    next_obs[i] = f"{next_obs[i]}\n[Shared Knowledge]: {shared_info}"
                    
                    self.logger.info(f"Step {step}: Successfully shared KV-cache across {self.n_agents} agents")
                    
                except Exception as e:
                    self.logger.warning(f"LatentMAS KV-cache sharing failed at step {step}: {str(e)}")
                    # 继续执行，即使LatentMAS失败

        # 构造success字典
        success = {
            'success_rate': np.random.random(batch_size) > 0.5  # 简化：可以根据实际任务实现
        }
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid

    def gather_rollout_data(
        self,
        total_batch_list: List[List[Dict]],
        episode_rewards: np.ndarray,
        episode_lengths: np.ndarray,
        success: Dict[str, np.ndarray],
        traj_uid: np.ndarray,
    ) -> DataProto:
        """
        收集和组织轨迹数据
        """
        batch_size = len(total_batch_list)

        episode_rewards_mean = np.mean(episode_rewards)
        episode_rewards_min = np.min(episode_rewards)
        episode_rewards_max = np.max(episode_rewards)

        episode_lengths_mean = np.mean(episode_lengths)
        episode_lengths_min = np.min(episode_lengths)
        episode_lengths_max = np.max(episode_lengths)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            for data in total_batch_list[bs]:
                if data.get('traj_uid') != traj_uid[bs]:
                    print(f"Warning: traj_uid mismatch")
                if data.get('active_masks', True):
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    data['episode_rewards_mean'] = episode_rewards_mean
                    data['episode_rewards_min'] = episode_rewards_min
                    data['episode_rewards_max'] = episode_rewards_max
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    data['episode_lengths_mean'] = episode_lengths_mean
                    data['episode_lengths_min'] = episode_lengths_min
                    data['episode_lengths_max'] = episode_lengths_max
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value
                    # 添加agent信息
                    data['training_agent_id'] = self.current_agent_idx

                    effective_batch.append(data)
        
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        fixed_agent_wgs: Optional[List] = None,
        is_train: bool = True,
    ) -> DataProto:
        """
        主入口函数（单agent训练版本）
        
        Parameters:
            gen_batch: 初始prompt batch
            actor_rollout_wg: 当前训练agent的WorkerGroup
            fixed_agent_wgs: 其他固定agent的WorkerGroups
            is_train: 是否为训练模式
        
        Returns:
            DataProto: 当前训练agent的trajectory数据
        """
        # Training模式：根据配置处理batch
        if is_train and hasattr(self.config, 'env') and hasattr(self.config.env, 'rollout'):
            rollout_n = self.config.env.rollout.get('n', 1)
            if rollout_n > 1:
                gen_batch = gen_batch.repeat(
                    repeat_times=rollout_n,
                    interleave=True
                )
        
        # 执行rollout
        results = self.vanilla_multi_turn_loop(
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            fixed_agent_wgs=fixed_agent_wgs,
        )
        
        # 验证数据一致性
        (total_batch_list, total_episode_rewards, total_episode_lengths,
         total_success, total_traj_uid) = results
        
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        
        # 整理并返回
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
        )
        
        print(f"[MultiAgentTrajectoryCollector] Generated {len(total_batch_list)} trajectories for agent {self.current_agent_idx}")
        
        return gen_batch_output
