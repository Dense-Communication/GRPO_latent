# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from verl.models.transformers.qwen2_vl import get_rope_index
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        # 训练 / 采样过程中用到的配置，包含数据长度、截断方式、环境设置等
        self.config = config
        # 用于对文本进行编码/解码的 tokenizer（如 Qwen / LLaMA 等）
        self.tokenizer = tokenizer
        # 多模态处理器（通常包含 image_processor 和 image_token 等属性）
        self.processor = processor

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        # 从上游的 gen_batch 中取出原始 prompt 和数据源标记
        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        # 取出 reward_model 和 extra_info（用于 naive reward manager）
        reward_model = gen_batch.non_tensor_batch.get('reward_model', None)
        if reward_model is not None:
            reward_model = reward_model[item]
        extra_info = gen_batch.non_tensor_batch.get('extra_info', None)
        if extra_info is not None:
            extra_info = extra_info[item]
        
        # 从环境返回的 obs 中拆出文本 / 图像 / anchor（锚点信息，用于 GiGPO）
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        # 每个 key 的 obs 都是 batch 级别的，因此这里根据 item 取出单条
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        # 是否多模态：只要当前样本存在图像就视为多模态
        is_multi_modal = obs_image is not None

        # anchor 信息可能是 Tensor，也可能是 Python 对象；统一转为 numpy/对象形式便于后续保存
        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # ---------------------- 构建对话结构（chat） ----------------------
        # 这里统一将环境 observation 封装为一个 "user" turn
        obs_content = ''
        if obs_text is not None:
            # 如果有文本观测，则直接使用
            obs_content += obs_text
        else:
            # 没有文本时给个 warning，方便调试（但对于无环境模式这是正常的）
            # print(f"Warning: No text observation found!")
            pass

        # 如果没有环境观测（obs_content 为空），则使用 raw_prompt（来自数据集）
        # 这样可以支持无环境模式下的训练（如 GSM8K 直接从数据集读取问题）
        if obs_content == '' and isinstance(raw_prompt, (list, np.ndarray)) and len(raw_prompt) > 0:
            # raw_prompt 已经是 chat 格式的列表
            chat = np.array(raw_prompt)
        else:
            # 有环境观测时，使用 obs_content 构建 chat
            chat = np.array([{
                "content": obs_content,
                "role": "user",
            }])
        
        # 使用 tokenizer 的 chat template 来生成最终的 prompt 字符串
        # add_generation_prompt=True 会在末尾附加模型的回答起始标记（如 <assistant> 等）
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # 用于存放当前样本的所有处理结果
        row_dict = {}
        
        # ---------------------- 多模态处理逻辑 ----------------------
        if is_multi_modal:
            # 首先用 vision token 占位符替换 prompt 中的 `<image>` 标记
            # raw_prompt 里后续会用来构建 raw_prompt_ids（不过内容被替换为 vision token 版本）
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            # 将当前样本的图像放入 multi_modal_data 中，供后续 image processor 使用
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            # 使用 processor.image_processor 将图像转为模型可接收的张量（grid 信息等）
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            # multi_modal_inputs 中一般会包含 pixel_values、image_grid_thw 等
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}

            # 如果存在 image_grid_thw，则需要根据 grid 大小替换 prompt 中的 `<image>` 为若干 vision token
            if image_grid_thw is not None:
                # merge_length 是每个合并后的 patch 所代表的原始格子数（如 2x2 -> 4）
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                # 依次找到所有 `<image>` 标记并替换为与视觉 token 数量一致的 placeholder
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                # 再把所有 `<|placeholder|>` 替换为模型定义的 image_token（如 <img>）
                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            # 非多模态场景，raw_prompt 直接使用 chat template 处理后的文本
            raw_prompt = prompt_with_chat_template
        
        # ---------------------- 文本 tokenization & padding ----------------------
        # 统一由 verl 封装的 tokenize_and_postprocess_data 完成编码、padding、截断等
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,  # 左填充：常见于右对齐的自回归模型
            truncation=self.config.data.truncation,  # 截断策略从配置中读取
        )
        
        # ---------------------- 位置编码（position_ids） ----------------------
        if is_multi_modal:
            # Qwen2-VL 的位置编码需要结合图像 grid 信息计算 rope index
            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask[0],
                )
              ]  # (1, 3, seq_len) 其中 3 维通常对应不同类型的 RoPE 位置
        else:
            # 纯文本情况下，使用通用的 compute_position_id_with_mask
            position_ids = compute_position_id_with_mask(attention_mask)

        # ---------------------- 构建 raw_prompt 的 token 序列 ----------------------
        # raw_prompt_ids 用于算法内部的一些对齐操作（如 KL、logprob 计算等）
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        # 如果 raw_prompt_ids 长度超过最大 prompt 长度，则根据配置选择截断策略
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                # 保留右半部分（截断左边）
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                # 保留左半部分（截断右边）
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                # 中间截断：保留两端信息
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                # 严格模式，长度超出则直接报错
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # ---------------------- 组装最终样本字典 ----------------------
        row_dict.update({
            'input_ids': input_ids[0],                # 模型输入 token id 序列
            'attention_mask': attention_mask[0],      # padding 掩码
            'position_ids': position_ids[0],          # 位置编码（RoPE index 等）
            'raw_prompt_ids': raw_prompt_ids,         # 原始 prompt 的 token 序列
            'anchor_obs': _obs_anchor,                # 观测对应的 anchor（GiGPO 使用）
            'index': item,                            # 在当前 batch 中的索引
            'data_source': data_source                # 数据来源标签（如 human / synthetic 等）
        })

        # 添加 reward_model 和 extra_info（用于 naive reward manager 计算规则奖励）
        if reward_model is not None:
            row_dict['reward_model'] = reward_model
        if extra_info is not None:
            row_dict['extra_info'] = extra_info

        # 配置项：是否保留原始聊天结构（方便调试或日志记录）
        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        # 当前 batch 的大小（即并行环境个数或并行样本个数）
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # 逐样本处理（可以后续并行化）
        for item in range(batch_size):
            # 针对第 item 个样本调用单样本预处理逻辑
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # 使用 collate_fn 将若干 dict 样本聚合成 batch 形式（统一 pad、stack 等）
        batch = collate_fn(processed_samples)
        
        # 将生成的 batch 封装回 DataProto，并保留原始 meta_info 信息
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        # total_batch_list 的长度 = 并行环境数量
        batch_size = len(total_batch_list)

        # 计算每个 success 指标的平均成功率（在所有 env 上取 mean）
        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        # 遍历每个 env 维度
        for bs in range(batch_size):
            # total_batch_list[bs] 是该环境完整的轨迹（按时间顺序的 step 列表）
            for data in total_batch_list[bs]:
                # 确保同一条轨迹中的 traj_uid 一致，避免数据错配
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                # active_masks 表明该步是否有效（未被提前截断或填充）
                if data['active_masks']:
                    # 为轨迹中的每条 step 记录终局信息（episode 级别）
                    # episode 总奖励
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode 总长度
                    data['episode_lengths'] = episode_lengths[bs]
                    # 工具调用次数
                    data['tool_callings'] = tool_callings[bs]
                    # success 指标（如 task_success, subgoal_success 等）
                    for key, value in success_rate.items():
                        data[key] = value

                    # 仅将有效的 step 加入最终的 effective_batch
                    effective_batch.append(data)
            
        # 将聚合好的 trajectory step 列表用 collate_fn 转成 batch 形式，再封装为 DataProto
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        # DataProto.batch 是一个 dict，长度通常等于并行环境数（或样本数）
        batch_size = len(gen_batch.batch)

        # 从环境中重置得到初始观测 obs 和 infos（支持传入 env_kwargs）
        env_kwargs = gen_batch.non_tensor_batch.pop('env_kwargs', None)
        # 如果 env_kwargs 为 None（无环境模式），创建一个 dummy list，长度等于 batch_size
        if env_kwargs is None:
            env_kwargs = [{} for _ in range(batch_size)]
        obs, infos = envs.reset(kwargs=env_kwargs)

        # obs 的长度应当与 gen_batch 一致（可能是 text list 或 image list）
        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        # ---------------------- 构建 uid_batch：用于环境分组（grouping） ----------------------
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                # 每 n 个样本共享一个 group uid，用于后续 filter_group_data 等操作
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)

        # is_done 标记每个环境是否结束（终止或截断）
        is_done = np.zeros(batch_size, dtype=bool)
        # 每个环境一个 traj_uid，用于区分不同环境的完整轨迹
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        # total_batch_list[i] 存储第 i 个环境在整个 rollout 过程中所有 step 的数据
        total_batch_list = [[] for _ in range(batch_size)]
        # total_infos[i] 存储第 i 个环境对应 step 的环境 info
        total_infos = [[] for _ in range(batch_size)]
        # 统计每个环境的 episode 步数与累计 reward
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        # 每个环境的工具调用总次数
        tool_callings = np.zeros(batch_size, dtype=np.float32)

        # ---------------------- 主 rollout 循环（多步交互） ----------------------
        for _step in range(self.config.env.max_steps):
            # active_masks 表明当前还没结束的环境
            active_masks = np.logical_not(is_done)

            # 将 gen_batch + 当前 obs 一起预处理成模型输入
            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            # 这些 key 是模型前向所需的；后续要 pop 出来喂给 actor
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            # raw_prompt_ids 不参与模型前向，属于非张量的一部分，也从 batch 中 pop 出
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            # 如果存在多模态数据 / raw_prompt / tools_kwargs，也要一道 pop 出给 actor
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")

            # batch.pop 会返回只包含这些 key 的 DataProto（作为模型输入），并从原 batch 中移除这些 key
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            # 将 meta_info 透传给模型侧，用于日志 / 策略区分等
            batch_input.meta_info = gen_batch.meta_info

            # ---------------------- Data Parallel padding ----------------------
            # 将 batch_input pad 到可以被 world_size 整除的大小，便于 DP 并行推理
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            # 调用 actor_rollout_wg 的 generate_sequences 生成动作（policy 输出）
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            # 去除之前为了整除而添加的 pad 数据
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            # 将 uid / traj_uid 等信息塞回 batch 的 non_tensor_batch 中，便于后续追踪轨迹归属
            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            # batch.union 会把模型输出（responses 等）合并回当前 batch
            batch = batch.union(batch_output)
            
            # 将模型生成的 token 序列 decode 为字符串，作为环境的文本 action
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            
            # 与环境交互：输入当前 step 的 actions，得到下一步 obs / reward / done / info
            next_obs, rewards, dones, infos = envs.step(text_actions)

            # 统一 reward 和 dones 的 shape（有些环境会多一维）
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            # 如果环境 info 中包含 is_action_valid（如非法动作检测），则一并保存
            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                # 默认都视为有效动作
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            # 工具调用统计：如果 info 中包含 tool_calling，则对 active 的环境进行累加
            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]

            # 只对 active 的环境累加 reward & step
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            # sanity check：环境必须返回与 batch_size 一致数量的 reward
            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            # 将 reward / active_masks 等也存进 batch.non_tensor_batch，便于后处理
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            # 将 DataProto 转成 list[dict]，每个 dict 对应一个环境当前 step 的完整信息
            batch_list: list[dict] = to_list_of_dict(batch)

            # 将本步数据累积到 total_batch_list / total_infos 中
            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # 更新 is_done：如果某个环境已经 done，就会被锁定为 True
            is_done = np.logical_or(is_done, dones)
                
            # 更新下一步的观测
            obs = next_obs

            # 如果所有环境都结束，则提前跳出循环
            if is_done.all():
                break
        
        # rollout 结束后，调用环境的 success_evaluator 统计成功指标
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        # 返回整个 vanilla rollout 的结果
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        # 动态采样模式下，会多次调用 vanilla_multi_turn_loop，直到收集到足够多的有效轨迹
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        # 已经尝试过的 rollout 次数
        try_count: int = 0
        # 最大尝试次数（避免死循环）
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        # 目标是收集的有效 step 数 >= train_batch_size * rollout.n
        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                # 打印当前已经收集的数量以及目标数量，便于调试
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            # 先进行一次 vanilla rollout，得到一个 batch 的轨迹
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            # 再使用 filter_group_data 对该 batch 的数据做筛选（过滤不符合某些条件的 group）
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(
                                                                                                batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            # 将筛选后的数据累积到全局列表中
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        # 多次 rollout 的 episode_rewards / lengths / uid / tool_callings 在 batch 维度上拼接
        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        # success 是一个 dict，每个 key 都是一个 np.ndarray，需要逐 key 拼接
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        # 训练阶段：通常会重复 gen_batch.n 次，并进行 interleave，提升并行采样效率
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
            
        # 根据配置选择动态采样（DAPO/动态 GiGPO）或普通采样
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )

        # 各个列表长度必须一致（每条轨迹都对应一个 reward/length/traj_uid/tool_callings）
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)
        
        # 将收集好的轨迹数据整理成最终的 DataProto，用于后续 RL 更新
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
        )
        
        return gen_batch_output
