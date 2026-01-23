# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""

本文件负责：
1) 解析 Hydra 配置，初始化/连接到 Ray 集群；
2) 通过 TaskRunner（Ray actor）在工作节点上构建训练所需的依赖：环境、分词器/处理器、
   资源池与各角色 Worker（Actor/Critic/RefPolicy/RM），以及奖励函数；
3) 创建数据集/采样器、TrajectoryCollector，并实例化 RayPPOTrainer；
4) 启动训练（trainer.fit()）。

注意：不与 `ray_trainer` 合并，是因为 `ray_trainer` 也被其它入口文件复用。
"""

import os  # 目前未直接使用；保留以兼容可能的外部依赖/环境变量读取

import hydra  # 配置管理
import ray  # 分布式执行引擎

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager  # 这里虽导入，但本文中采用自定义 EpisodeRewardManager


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Hydra 入口函数。

    Hydra 会根据 `config/ppo_trainer.yaml` 加载配置并注入到 `config`。
    """
    run_ppo(config)


def run_ppo(config) -> None:
    """在（本地）Ray 集群上运行 PPO 训练流程。"""
    import os
    # 在进程内设置环境变量，避免使用 runtime_env（离线环境可能超时）
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARN")
    os.environ.setdefault("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "true")
    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")

    if not ray.is_initialized():
        # 如果用户没有在外部预先 `ray start`，这里会在本机启动一个本地 Ray 集群。
        # 注意：离线环境不使用 runtime_env，避免 runtime_env_agent 超时
        # 限制 CPU 数量避免 "Resource temporarily unavailable" 错误
        num_cpus = config.ray_init.num_cpus
        if num_cpus is None:
            num_cpus = min(os.cpu_count() or 8, 32)  # 限制最大 32 个 CPU
        # Ray 临时目录：优先使用环境变量 RAY_TMPDIR
        ray_tmpdir = os.environ.get("RAY_TMPDIR", "/tmp/ray")
        os.makedirs(ray_tmpdir, exist_ok=True)

        ray.init(
            num_cpus=num_cpus,
            include_dashboard=False,  # 离线环境禁用 dashboard
            _temp_dir=ray_tmpdir,  # 指定临时目录
        )

    # 使用一个独立的 Ray Actor 在 worker 节点运行主流程，避免被调度到 head（常见做法：节省/隔离资源）
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # 提示：请确保 main_task 不要调度在 head 节点
class TaskRunner:
    """真正执行训练准备与启动的 Ray Actor。

    将主要逻辑放入 Ray Actor 内有助于：
    - 在工作节点使用其本地 GPU/CPU 资源；
    - 保持 head 节点轻负载；
    - 统一数据/模型从远端存储拉取到本地的路径环境。
    """

    def run(self, config):
        # 1) 打印解析后的配置，便于调试/复现
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True: 解析 ${var} 之类的符号引用
        OmegaConf.resolve(config)  # 就地解析，后续使用更安全

        # 2) 将模型 checkpoint 从远端（如 HDFS/OSS）下载到本地或 /dev/shm
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        # 3) 创建训练环境与验证环境
        # - 这里优先从 agent_system.environments.make_envs 构建；
        # - 若用户在配置中显式关闭环境（env_name 为空/none/off），则提供一个最小 Dummy 环境兜底，
        #   以便仍可进行基于函数式 RM（无环境交互）的 PPO。
        from agent_system.environments import make_envs

        # 读取环境名称（可能为空）
        env_name = str(getattr(getattr(config, "env", None), "env_name", "") or "").strip()

        def _build_dummy_vec_env():
            """构造一个极简的“向量环境”以兼容训练流程的接口。

            - reset 返回占位 obs/infos，长度与 batch 对齐；
            - step 立即 done，reward=0.0；真正的奖励由 RM 负责计算。
            """

            class _DummyVecEnv:
                def reset(self, kwargs=None):
                    # kwargs 应该是一个字典列表，每个环境对应一个字典
                    # 如果 kwargs 为 None，则无法确定 batch size，这种情况不应该发生
                    if kwargs is None:
                        # 如果没有 kwargs，使用默认 batch size 1
                        # 注意：这只是一个兜底方案，实际使用时应该传入 kwargs
                        bs = 1
                    elif isinstance(kwargs, list):
                        # kwargs 是字典列表，长度就是 batch size
                        bs = len(kwargs)
                    elif isinstance(kwargs, dict):
                        # 旧版本兼容：如果是单个字典，尝试从中提取 batch_size
                        bs = kwargs.get("batch_size", kwargs.get("n", 1)) or 1
                    else:
                        bs = 1
                    # 返回字典格式的 obs，与 rollout_loop.py:343 的期望一致
                    return {'text': [None] * bs, 'image': None}, [{} for _ in range(bs)]

                def step(self, actions):
                    import numpy as np
                    bs = len(actions) if actions is not None else 1
                    # step 也应该返回字典格式的 obs
                    obs = {'text': [None] * bs, 'image': None}
                    # rewards 和 dones 应该是 numpy 数组，不是列表
                    rews = np.zeros(bs, dtype=np.float32)
                    dones = np.ones(bs, dtype=bool)
                    infos = [{} for _ in range(bs)]
                    return obs, rews, dones, infos

                def success_evaluator(self, *args, **kwargs):
                    """评估任务成功率。对于 Dummy 环境，返回空的成功率数组。"""
                    import numpy as np
                    total_batch_list = kwargs.get('total_batch_list', [])
                    batch_size = len(total_batch_list)
                    # 返回全零的成功率数组（因为 Dummy 环境没有实际任务）
                    return {'success_rate': np.zeros(batch_size, dtype=np.float32)}

            return _DummyVecEnv()

        # 选择真实环境或 Dummy 环境
        if env_name == "" or env_name.lower() in ("none", "null", "off"):
            envs = _build_dummy_vec_env()
            val_envs = _build_dummy_vec_env()
        else:
            envs, val_envs = make_envs(config)

        # 4) 实例化 tokenizer 与（可选）processor（多模态时使用）
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # use_fast=True：若模型为多模态 LLM 时，processor 可能非空；否则为 None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # 5) vLLM 版本前置校验（避免运行期才报不支持）
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            # 在 vLLM 0.7.3 之前不支持 PPO + LoRA 的组合
            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # 6) 根据并行策略选择 Worker 实现 & WorkerGroup 类型
        # - FSDP/FSDP2：使用 FSDP worker 与 RayWorkerGroup
        # - Megatron：使用 Megatron worker 与 NVMegatronRayWorkerGroup
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import (
                ActorRolloutRefWorker,
                AsyncActorRolloutRefWorker,
                CriticWorker,
            )

            # rollout.mode 决定 actor 是否异步
            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError  # 其它并行策略暂不支持

        # 7) 构建角色到 Ray 远程类的映射 + 资源池
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # 角色 -> Worker 类（Ray remote 包装）
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        # 定义一个全局资源池：每个节点固定 n_gpus_per_node 个 GPU 槽位
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        # 将不同角色映射到同一资源池（简单起见，可扩展为多池）
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # 8) 可选：奖励模型（RM）角色
        #    - 规则 RM：直接计算分数
        #    - 模型 RM：调用模型推理
        #    - 代码题：可送沙箱跑测试
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # 9) 可选：参考策略（RefPolicy），用于 KL 效用或 KL loss
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # 10) 奖励管理器
        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == 'episode':
            # EpisodeRewardManager: 用于 agent 环境交互，从 episode_rewards 获取分数
            from agent_system.reward_manager import EpisodeRewardManager
            reward_manager_cls = EpisodeRewardManager
            # 训练时：不做长度归一与额外多轮检查；验证时：num_examine=1 做一次更严格的核验
            reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
            val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)
        elif reward_manager_name == 'naive':
            # NaiveRewardManager: 用于规则奖励（如 GSM8K），通过 compute_score 函数计算分数
            from verl.trainer.ppo.reward import load_reward_manager
            reward_fn = load_reward_manager(config, tokenizer, num_examine=0)
            val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1)
        else:
            raise NotImplementedError(f"Unknown reward_manager: {reward_manager_name}")

        # 11) 构建资源池管理器（描述“池子规模”与“角色到池”的映射）
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # 12) 延伸说明：在 verl 主线下，actor_rollout_ref.rollout.n>1 表示 GRPO；
        #     在本项目（verl+env）中保持 n=1，通过 env.rollout.n 实现 GRPO 的并行分支
        assert config.actor_rollout_ref.rollout.n == 1, (
            "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"
        )

        # 13) 轨迹采集器：负责与环境多轮交互、拼接一条完整轨迹（包含 obs/prompt、action/response 等）
        from agent_system.multi_turn_rollout import TrajectoryCollector
        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        # 14) 构建数据集与采样器
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # 15) 实例化训练器
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
        )

        # 16) 启动各 Worker，并进入训练循环
        trainer.init_workers()
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """根据配置创建 RL 数据集。

    Args:
        data_paths: 训练或验证数据文件列表。
        data_config: 数据相关配置（shuffle、最大长度、字段映射等）。
        tokenizer: 文本分词器（HuggingFace）。
        processor: 多模态处理器（如图像/视频/音频等，可能为 None）。

    Returns:
        dataset: 继承自 `torch.utils.data.Dataset` 的数据集实例。
    """
    from torch.utils.data import Dataset
    from verl.utils.dataset.rl_dataset import RLHFDataset

    # 若用户提供自定义 Dataset 类（路径+类名），则动态加载；否则默认使用 RLHFDataset
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """为数据集创建采样器（更好地支持断点续训）。

    - 当 `shuffle=True` 时，使用固定随机种子的 `RandomSampler`；
    - 否则使用 `SequentialSampler`。
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # 使用 Sampler 便于在恢复训练时重现采样顺序
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
