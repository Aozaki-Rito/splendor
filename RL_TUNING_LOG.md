# RL 调参记录

最后更新：2026-04-09

## 目标

当前主目标不是单纯提高胜率，而是把 RL 在真实对局中的节奏压得更快。

- 目标指标：`avg_rl_trigger_round <= 25`
- 含义：RL 首次达到 `15+` 分时，UI 左上角显示的平均回合数
- 当前最好结果：`31.0`
- 当前最好模型：[runs/rl/v9a_ppo_random_seed7_t100k_step002_gamma095_speed15_ref25/model.zip](/Users/bytedance/Desktop/splendor/runs/rl/v9a_ppo_random_seed7_t100k_step002_gamma095_speed15_ref25/model.zip)

## 评估口径

所有对比应尽量使用同一套 benchmark 口径，避免只看训练日志里的 `ep_len_mean`。

- 脚本：[scripts/benchmark_rl_model.py](/Users/bytedance/Desktop/splendor/scripts/benchmark_rl_model.py)
- 对手：`random`
- RL 座位：`2`
- 局数：`10`
- 种子范围：`3000-3009`

核心指标解释：

- `avg_rl_trigger_round`：RL 首次到 `15+` 分的平均 UI 回合数；这是当前最重要指标
- `avg_final_round`：整局真正结束时的平均 UI 回合数
- `win_rate`：RL 胜率
- `avg_total_actions`：整局总动作数，作为辅助节奏指标

推荐命令：

```bash
MPLCONFIGDIR=.mplconfig .venv/bin/python scripts/benchmark_rl_model.py \
  --model-path runs/rl/<run_name>/model.zip \
  --episodes 10 \
  --seed-start 3000 \
  --opponent random \
  --rl-seat 2 \
  --output runs/rl/<run_name>/benchmark_random_seat2_ep10_seed3000.json
```

## 当前主线

目前已经验证过的方向里，最值得继续的是：

- 固定对手为 `random`
- 奖励保留 `step_penalty + win_speed_scale`
- 在这个奖励组合下继续调 PPO 超参数

目前已基本放弃的方向：

- `rule_based` 作为训练对手：训练日志变好看，但真实 benchmark 更慢
- `round_penalty_scale`：帮助不明显
- `score_speed_scale`：训练容易被带歪

## 已完成实验

说明：

- 表中 `结论` 是相对当前最好基线 `v9a` 而言
- 未特别说明时，`win_speed_scale=1.5`、`win_speed_reference_round=25`
- 结果以真实 benchmark 为准，不以训练期 `ep_len_mean` 为准

| run_name | 对手 | timesteps | lr | n_steps | batch_size | gamma | step_penalty | avg_rl_trigger_round | avg_final_round | win_rate | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `v9a_ppo_random_seed7_t100k_step002_gamma095_speed15_ref25` | random | 100000 | 3e-4 | 1024 | 256 | 0.95 | 0.02 | **31.0** | **32.3** | 0.90 | 当前最好基线 |
| `v12a_ppo_rule_based_seed7_t100k_step002_gamma095_speed15_ref25` | rule_based | 100000 | 3e-4 | 1024 | 256 | 0.95 | 0.02 | 33.6 | 34.6 | 1.00 | 更强对手训练未带来收益 |
| `v12c_ppo_rule_based_seed7_t100k_step005_gamma095_speed15_ref25` | rule_based | 100000 | 3e-4 | 1024 | 256 | 0.95 | 0.05 | 33.0 | 34.0 | 0.90 | 仍慢于 `v9a` |
| `v14a_ppo_random_seed7_t300k_step002_gamma095_speed15_ref25` | random | 300000 | 3e-4 | 1024 | 256 | 0.95 | 0.02 | 32.1 | 33.1 | 1.00 | 单纯拉长训练无收益 |
| `v14b_ppo_random_seed7_t300k_step002_gamma095_speed15_lr2e4_ns512_bs128_ref25` | random | 300000 | 2e-4 | 512 | 128 | 0.95 | 0.02 | 34.6 | 35.5 | 1.00 | 明显退化 |
| `v15a_ppo_random_seed7_t100k_lr2e4_ns1024_bs256_gamma095_step002_speed15_ref25` | random | 100000 | 2e-4 | 1024 | 256 | 0.95 | 0.02 | 33.3 | 34.3 | 1.00 | 降低学习率无收益 |
| `v15b_ppo_random_seed7_t100k_lr3e4_ns1024_bs256_gamma095_step001_speed15_ref25` | random | 100000 | 3e-4 | 1024 | 256 | 0.95 | 0.01 | 31.9 | 32.9 | 1.00 | 接近基线，但仍未超过 `v9a` |
| `v15c_ppo_random_seed7_t100k_lr3e4_ns1024_bs256_gamma090_step001_speed15_ref25` | random | 100000 | 3e-4 | 1024 | 256 | 0.90 | 0.01 | 33.6 | 34.6 | 1.00 | `gamma=0.90` 明显变慢 |
| `v15d_ppo_random_seed7_t100k_lr3e4_ns1024_bs256_gamma095_step0005_speed15_ref25` | random | 100000 | 3e-4 | 1024 | 256 | 0.95 | 0.005 | 33.2 | 34.2 | 1.00 | 惩罚太轻也变慢 |
| `v16a_ppo_random_seed7_t100k_lr3e4_ns1024_bs256_gamma095_ent0005_step002_speed15_ref25` | random | 100000 | 3e-4 | 1024 | 256 | 0.95 | 0.02 | 32.67 | 33.6 | 0.90 | `ent_coef=0.005` 无收益 |
| `v16b_ppo_random_seed7_t100k_lr3e4_ns1024_bs256_gamma095_ent001_step002_speed15_ref25` | random | 100000 | 3e-4 | 1024 | 256 | 0.95 | 0.02 | 32.25 | 33.8 | 0.80 | 个别样本很快，但整体不稳 |
| `v17a_ppo_random_seed7_t100k_lr3e4_ns1024_bs256_gamma095_ent0001_step002_speed15_ref25` | random | 100000 | 3e-4 | 1024 | 256 | 0.95 | 0.02 | 33.8 | 34.8 | 1.00 | 更小熵正则仍无收益 |
| `v17b_ppo_random_seed7_t100k_lr3e4_ns1024_bs256_gamma095_ent0002_step002_speed15_ref25` | random | 100000 | 3e-4 | 1024 | 256 | 0.95 | 0.02 | 33.6 | 34.5 | 1.00 | 更小熵正则仍无收益 |
| `v18a_ppo_random_seed7_t100k_lr3e4_ns512_bs256_gamma095_step002_speed15_ref25` | random | 100000 | 3e-4 | 512 | 256 | 0.95 | 0.02 | 33.25 | 33.9 | 0.80 | 单独缩短 rollout 明显退化 |
| `v18b_ppo_random_seed7_t100k_lr3e4_ns1024_bs128_gamma095_step002_speed15_ref25` | random | 100000 | 3e-4 | 1024 | 128 | 0.95 | 0.02 | 32.3 | 33.3 | 1.00 | 单独减小 batch 仍未超过基线 |
| `v19a_ppo_random_seed7_t100k_lr3e4_ns1024_bs256_gamma095_gae090_step002_speed15_ref25` | random | 100000 | 3e-4 | 1024 | 256 | 0.95 | 0.02 | 32.7 | 33.6 | 1.00 | `gae_lambda=0.90` 无收益 |
| `v19b_ppo_random_seed7_t100k_lr3e4_ns1024_bs256_gamma095_clip015_step002_speed15_ref25` | random | 100000 | 3e-4 | 1024 | 256 | 0.95 | 0.02 | 31.33 | 32.8 | 0.90 | 接近基线，值得继续沿 `clip_range` 附近挖 |
| `v20a_ppo_random_seed7_t100k_lr3e4_ns1024_bs256_gamma095_clip010_step002_speed15_ref25` | random | 100000 | 3e-4 | 1024 | 256 | 0.95 | 0.02 | 32.1 | 33.1 | 1.00 | `clip_range=0.10` 无收益 |
| `v20b_ppo_random_seed7_t100k_lr3e4_ns1024_bs256_gamma095_clip015_step003_speed15_ref25` | random | 100000 | 3e-4 | 1024 | 256 | 0.95 | 0.03 | 33.22 | 34.1 | 0.90 | 惩罚增大到 `0.03` 退化 |
| `v21a_ppo_random_seed7_t100k_lr3e4_ns1024_bs256_gamma095_clip010_step003_speed15_ref25` | random | 100000 | 3e-4 | 1024 | 256 | 0.95 | 0.03 | 32.5 | 33.5 | 1.00 | `clip=0.10 + step=0.03` 仍无收益 |
| `v21b_ppo_random_seed7_t100k_lr3e4_ns1024_bs256_gamma095_clip0125_step002_speed15_ref25` | random | 100000 | 3e-4 | 1024 | 256 | 0.95 | 0.02 | 33.5 | 35.0 | 0.80 | `clip_range=0.125` 明显变差 |
| `v22a_ppo_random_seed7_t100k_lr3e4_ns1024_bs256_gamma095_clip0125_step003_speed15_ref25` | random | 100000 | 3e-4 | 1024 | 256 | 0.95 | 0.03 | 31.7 | 32.7 | 1.00 | 这一批里最好，但仍未超过 `v9a` |
| `v22b_ppo_random_seed7_t100k_lr3e4_ns1024_bs256_gamma095_clip015_step0025_speed15_ref25` | random | 100000 | 3e-4 | 1024 | 256 | 0.95 | 0.025 | 32.3 | 33.3 | 1.00 | 中等惩罚也未超过基线 |

## 仍在运行的实验

| run_name | 状态 | 对手 | timesteps | lr | n_steps | batch_size | gamma | step_penalty | 目的 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
当前没有正在跑、且尚未 benchmark 的实验。

## 当前判断

截至目前，比较明确的结论有十八条：

1. `rule_based` 训练对手没有带来目标指标收益，因此当前不再继续沿这条线扩实验。
2. 单纯把训练从 `100k` 拉到 `300k` 没有改善终局节奏，说明问题不是“训练步数不够”这么简单。
3. 把 `lr` 从 `3e-4` 降到 `2e-4` 没有收益。
4. 把 `step_penalty` 从 `0.02` 调到 `0.01` 后，结果接近基线，但仍未超过 `v9a`。
5. 把 `gamma` 从 `0.95` 压到 `0.90` 会明显恶化终局节奏。
6. 把 `step_penalty` 继续减到 `0.005` 也会退化，说明动作惩罚不能太轻。
7. `ent_coef=0.005` 没有带来收益。
8. `ent_coef=0.01` 会出现极快样本，但整体不稳定，胜率下降到 `0.80`。
9. `ent_coef=0.001` 和 `0.002` 也没有收益，说明当前问题大概率不在熵正则。
10. 当前最强模型仍然是 `v9a`，后续应优先换到其它 PPO 本体参数，而不是继续扫 `ent_coef`。
11. 单独把 `n_steps` 从 `1024` 降到 `512` 会明显退化。
12. 单独把 `batch_size` 从 `256` 降到 `128` 也没有超过基线。
13. `gae_lambda=0.90` 没有带来收益。
14. `clip_range=0.15` 是目前除 `v9a` 外最接近基线的一条线，值得继续沿这一侧细挖。
15. `clip_range=0.10` 没有带来收益。
16. `step_penalty` 提高到 `0.03` 会整体退化。
17. `clip_range=0.125` 也没有带来收益。
18. 这一轮里最好的 `v22a` 也只有 `31.7`，仍然没有超过 `v9a` 的 `31.0`。

当前最值得继续验证的是：

- `gamma` 先固定在 `0.95`
- `step_penalty` 先回到当前最优的 `0.02`
- 停止继续搜索 `ent_coef`
- `n_steps=1024 / batch_size=256` 暂时继续作为基准组合
- 暂停继续扫 `gae_lambda`
- `clip_range + step_penalty` 这一组也已经做过一轮系统搜索，但没有打破 `v9a`
- 结论：纯调 PPO 超参数这条线目前没有把目标推进到 `25` 回合以内，下一步需要换方案

## 文档维护约定

后续每次有新实验结果，至少补这三件事：

1. 在“已完成实验”或“仍在运行的实验”表中加一行
2. 如果 benchmark 做完，把 `avg_rl_trigger_round / avg_final_round / win_rate` 补齐
3. 在“当前判断”里更新一句结论，不让文档只堆数据、不形成判断
