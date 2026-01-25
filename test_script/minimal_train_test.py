#!/usr/bin/env python3
"""
最小化训练验证脚本 - 验证:
1. 模型能正常加载
2. 奖励函数能正常工作
3. 梯度能正常更新模型参数
4. 训练后模型输出有变化
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 离线模式
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_PATH = "/p/scratch/westai0052/liu52/models/Qwen2.5-1.5B-Instruct"

# GSM8K 测试题目
TEST_PROMPTS = [
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
]
EXPECTED_ANSWERS = ["18", "3"]

def compute_reward(response: str, expected: str) -> float:
    """简单的奖励函数：答案正确得1分，否则0分"""
    import re
    # 提取数字
    numbers = re.findall(r'-?\d+\.?\d*', response)
    if numbers and numbers[-1] == expected:
        return 1.0
    return 0.0

def generate_response(model, tokenizer, prompt: str, max_new_tokens=128) -> str:
    """生成模型回复"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # 使用贪婪解码避免概率分布问题
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 贪婪解码
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

def compute_loss_with_reward(model, tokenizer, prompt: str, response: str, reward: float):
    """基于奖励计算损失并返回"""
    # 构建完整对话
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    # 前向传播
    outputs = model(**inputs, labels=inputs["input_ids"])

    # 用奖励加权损失 (REINFORCE 风格)
    # 高奖励 -> 低损失（鼓励这种输出）
    # 低奖励 -> 高损失（抑制这种输出）
    weighted_loss = outputs.loss * (1.0 - reward)

    return weighted_loss, outputs.loss.item()

def main():
    print("=" * 60)
    print("最小化 GSM8K 训练验证")
    print("=" * 60)

    # 1. 加载模型和分词器
    print("\n[1/5] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"✓ 模型加载成功: {MODEL_PATH}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 2. 训练前测试
    print("\n[2/5] 训练前评估...")
    pre_train_responses = []
    pre_train_rewards = []

    for prompt, expected in zip(TEST_PROMPTS, EXPECTED_ANSWERS):
        response = generate_response(model, tokenizer, prompt)
        reward = compute_reward(response, expected)
        pre_train_responses.append(response)
        pre_train_rewards.append(reward)
        print(f"  问题: {prompt[:50]}...")
        print(f"  回答: {response[:100]}...")
        print(f"  奖励: {reward}")

    print(f"\n训练前平均奖励: {sum(pre_train_rewards)/len(pre_train_rewards):.2f}")

    # 3. 记录训练前的部分参数值
    print("\n[3/5] 记录训练前参数...")
    # 获取第一个可训练层的参数
    first_param = None
    param_name = None
    for name, param in model.named_parameters():
        if param.requires_grad and 'weight' in name:
            first_param = param.data.clone()
            param_name = name
            break
    print(f"  监控参数: {param_name}")
    print(f"  参数形状: {first_param.shape}")
    print(f"  参数均值(训练前): {first_param.mean().item():.6f}")

    # 4. 简单训练循环
    print("\n[4/5] 开始训练 (3轮)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    model.train()
    for epoch in range(3):
        total_loss = 0
        for prompt, expected in zip(TEST_PROMPTS, EXPECTED_ANSWERS):
            # 生成回复
            model.eval()
            with torch.no_grad():
                response = generate_response(model, tokenizer, prompt)
            model.train()

            # 计算奖励
            reward = compute_reward(response, expected)

            # 计算损失并更新
            optimizer.zero_grad()
            loss, raw_loss = compute_loss_with_reward(model, tokenizer, prompt, response, reward)
            loss.backward()
            optimizer.step()

            total_loss += raw_loss

        avg_loss = total_loss / len(TEST_PROMPTS)
        print(f"  Epoch {epoch+1}: 平均损失 = {avg_loss:.4f}")

    # 5. 验证参数已更新
    print("\n[5/5] 验证训练效果...")

    # 检查参数变化
    for name, param in model.named_parameters():
        if name == param_name:
            new_param = param.data
            diff = (new_param - first_param).abs().mean().item()
            print(f"  参数变化量: {diff:.8f}")
            if diff > 0:
                print("  ✓ 模型参数已更新!")
            else:
                print("  ✗ 警告: 参数未变化")
            break

    # 训练后评估
    print("\n训练后评估...")
    model.eval()
    post_train_rewards = []

    for i, (prompt, expected) in enumerate(zip(TEST_PROMPTS, EXPECTED_ANSWERS)):
        response = generate_response(model, tokenizer, prompt)
        reward = compute_reward(response, expected)
        post_train_rewards.append(reward)
        print(f"  问题 {i+1}:")
        print(f"    训练前: {pre_train_responses[i][:80]}...")
        print(f"    训练后: {response[:80]}...")
        print(f"    奖励: {pre_train_rewards[i]} -> {reward}")

    print("\n" + "=" * 60)
    print("训练验证总结")
    print("=" * 60)
    print(f"训练前平均奖励: {sum(pre_train_rewards)/len(pre_train_rewards):.2f}")
    print(f"训练后平均奖励: {sum(post_train_rewards)/len(post_train_rewards):.2f}")
    print(f"参数更新: {'✓ 成功' if diff > 0 else '✗ 失败'}")
    print("=" * 60)

if __name__ == "__main__":
    main()
