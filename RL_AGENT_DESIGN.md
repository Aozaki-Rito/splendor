# Splendor RL Agent Design

这份文档用于整理“璀璨宝石自主决策 Agent”的强化学习设计。

目标不是继续依赖 LLM prompt，而是把当前项目中的规则引擎包装成一个适合 RL 的环境，并明确定义：

- 状态空间 `observation space`
- 动作空间 `action space`
- 合法动作掩码 `action mask`
- 后续环境接口与训练方向

本文档是工作草案，会随着讨论持续修订。

## 1. 目标

希望训练出一个能够在璀璨宝石环境中自主决策的 Agent。

这里的“自主决策”指：

- 不依赖文本 prompt 推理
- 不依赖手写启发式直接选动作
- 使用神经网络策略根据结构化状态直接输出动作分布

第一阶段目标：

- 基于当前规则引擎构建 RL 训练接口
- 先支持 2 人对局
- 先支持固定长度 observation
- 先支持固定编号动作空间 + action mask

## 2. 当前项目可直接复用的能力

当前代码已经具备 RL 所需的核心部件：

- 合法动作枚举：[game/game.py](/home/aozaki/projects/code/splendor/game/game.py)
  - `Game.get_valid_actions()`
- 动作执行：[game/game.py](/home/aozaki/projects/code/splendor/game/game.py)
  - `Game.execute_action(action)`
- 全局游戏状态导出：[game/game.py](/home/aozaki/projects/code/splendor/game/game.py)
  - `Game.get_game_state()`
- 玩家状态导出：[game/player.py](/home/aozaki/projects/code/splendor/game/player.py)
  - `Player.to_dict()`
- 棋盘状态导出：[game/board.py](/home/aozaki/projects/code/splendor/game/board.py)
  - `Board.to_dict()`

## 3. 设计原则

状态空间与动作空间设计遵循以下原则：

1. 状态必须是固定长度向量。
2. 动作必须是固定编号的离散动作。
3. 当前步不可执行的动作通过 `action_mask` 屏蔽。
4. 不暴露真实对局中不可见的信息。
5. 预留卡 `reserved_cards` 是公开信息，可以编码进状态。
6. 未翻开的牌堆内容不应暴露，只能保留牌堆剩余数量。
7. observation 最终建议编码为 `float32` 向量。

## 4. 动作空间

采用固定编号的离散动作空间，而不是每回合动态重建动作 ID。

这样可以方便：

- PPO / A2C / DQN 等算法直接输出固定维度动作分布
- 使用统一 `action_mask`

### 4.1 动作总数

当前建议的固定动作总数为 `45`。

### 4.2 动作编号

#### A. 拿取三种不同颜色宝石 `0-9`

固定顺序如下：

- `0`: `WBG`
- `1`: `WBR`
- `2`: `WBK`
- `3`: `WGR`
- `4`: `WGK`
- `5`: `WRK`
- `6`: `BGR`
- `7`: `BGK`
- `8`: `BRK`
- `9`: `GRK`

其中：

- `W` = white
- `B` = blue
- `G` = green
- `R` = red
- `K` = black

#### B. 拿取两个同色宝石 `10-14`

- `10`: `WW`
- `11`: `BB`
- `12`: `GG`
- `13`: `RR`
- `14`: `KK`

#### C. 预留展示区卡牌 `15-26`

按固定槽位编码：

- `15-18`: 预留 1 级展示区第 `0-3` 张
- `19-22`: 预留 2 级展示区第 `0-3` 张
- `23-26`: 预留 3 级展示区第 `0-3` 张

#### D. 预留牌堆顶 `27-29`

- `27`: 预留 1 级牌堆顶
- `28`: 预留 2 级牌堆顶
- `29`: 预留 3 级牌堆顶

#### E. 购买展示区卡牌 `30-41`

- `30-33`: 购买 1 级展示区第 `0-3` 张
- `34-37`: 购买 2 级展示区第 `0-3` 张
- `38-41`: 购买 3 级展示区第 `0-3` 张

#### F. 购买预留卡 `42-44`

- `42`: 购买自己预留区第 `0` 张
- `43`: 购买自己预留区第 `1` 张
- `44`: 购买自己预留区第 `2` 张

### 4.3 action mask

环境每一步都提供一个 `45` 维 `action_mask`：

- 可执行动作：`1`
- 不可执行动作：`0`

策略网络输出 45 维 logits 后，只在 mask 允许的动作中采样或取最大值。

## 5. 状态空间

状态空间采用“当前玩家视角”的固定长度 observation。

说明：

- 当前玩家始终编码成 `self`
- 对手编码成 `opponent`
- 不暴露未翻开的牌堆具体内容
- 预留卡是公开信息，因此允许编码

## 6. observation 结构

当前建议 observation 由以下部分组成：

1. 全局特征
2. 我方特征
3. 对手特征
4. 公共展示卡特征
5. 我方预留卡特征
6. 贵族特征
7. 牌堆剩余数量

## 7. 全局特征

建议包含：

- `round_number`
- `last_round`
- `num_players`

建议：

- `round_number` 做归一化
- `last_round` 用 `0/1`
- `num_players` 当前可固定为 2 人版本，后续再扩展

## 8. 我方特征

当前建议保留：

- `score`
- `gems[6]`
  - `white blue green red black gold`
- `discounts[5]`
  - `white blue green red black`
- `reserved_count`
- `nobles_count`
- `gem_slots_left`

说明：

- `gem_slots_left` 保留
- `total_gems` 不保留，因为与 `gem_slots_left` 信息重合
- `owned_cards_count` 暂不保留，因为价值不大，且大部分信息已被分数与折扣覆盖

## 9. 对手特征

在 2 人局中，编码 1 个对手。

当前建议包含：

- `score`
- `gems[6]`
- `discounts[5]`
- `reserved_count`
- `nobles_count`

说明：

- 对手的 `reserved_cards` 是公开信息，因此建议单独编码
- 第一版先不额外保留对手 `gem_slots_left`

## 10. 公共展示卡特征

展示区共有 12 个固定槽位：

- 1 级 4 张
- 2 级 4 张
- 3 级 4 张

每张卡编码为固定长度向量。

### 10.1 单张卡特征

当前版本采用以下字段：

- `exists` `1`
- `level_onehot` `3`
- `points` `1`
- `bonus_color_onehot` `5`
- `cost` `5`
- `buyable_now` `1`
- `missing_after_discount` `5`

合计：`21` 维

### 10.2 各字段含义

#### `exists`

表示该槽位是否有卡：

- 有卡：`1`
- 空槽：`0`

#### `level_onehot`

卡牌等级 one-hot：

- 1 级：`[1,0,0]`
- 2 级：`[0,1,0]`
- 3 级：`[0,0,1]`

#### `points`

该卡的分值，建议做归一化。

#### `bonus_color_onehot`

购买该卡后获得的永久折扣色，用 5 维 one-hot 表示：

- white
- blue
- green
- red
- black

#### `cost`

该卡的原始购买成本，共 5 维：

- white
- blue
- green
- red
- black

#### `buyable_now`

当前玩家此刻是否可以直接购买该卡：

- 可以：`1`
- 不可以：`0`

#### `missing_after_discount`

当前玩家在**只考虑永久折扣**的情况下，购买该卡还缺多少颜色资源。

这里已经明确：

- 只考虑折扣
- 不把当前手里的宝石数量计入该字段

也就是说，它表达的是：

- 当前引擎距离这张卡还有多远

而不是：

- 我这回合距离立刻买下它还差多少宝石

## 11. 我方预留卡特征

我方最多有 3 张预留卡。

建议使用和公共展示卡相同的编码方式：

- 每张也是 `21` 维
- 一共 3 个固定槽位

如果某个槽位为空，则：

- `exists = 0`
- 其余维度填 0

## 12. 对手预留卡特征

对手的预留卡也是公开信息，因此建议显式编码。

在 2 人局中，当前建议：

- 为对手保留 3 个固定预留卡槽位
- 每个槽位的编码方式与“公共展示卡 / 我方预留卡”一致
- 如果某个槽位为空，则整张卡向量全 0

也就是说，对手预留卡同样使用如下单卡编码：

- `exists` `1`
- `level_onehot` `3`
- `points` `1`
- `bonus_color_onehot` `5`
- `cost` `5`
- `buyable_now` `1`
- `missing_after_discount` `5`

合计：

- 每张 `21` 维
- 3 张共 `63` 维

说明：

- 这里的 `buyable_now` 和 `missing_after_discount` 默认仍然以“当前玩家视角”计算
- 也就是说，这些特征表达的是“这张公开预留卡对我来说是否可买、距离我还有多远”
- 第一版先不额外加入“对手视角的 buyable_now / missing_after_discount”

## 13. 贵族特征

2 人局中通常有 3 个贵族槽位。

建议每个贵族编码包含：

- `exists`
- `points`
- `requirements[5]`
- `missing_for_me[5]`
- `missing_for_opponent[5]`

说明：

- `missing_for_me` 表示我方距离满足该贵族要求还差多少折扣
- `missing_for_opponent` 表示对手距离满足该贵族要求还差多少折扣

## 14. 牌堆信息

牌堆内部内容不可见，因此不能暴露。

当前只允许保留：

- 1 级剩余数量
- 2 级剩余数量
- 3 级剩余数量

即：

- `deck_counts[3]`

## 15. observation 编码类型

虽然很多特征语义上是整数或离散值，但 observation 整体建议编码为连续实值向量。

推荐：

- 最终全部转换为 `float32`
- 布尔值用 `0/1`
- one-hot 用 `0/1`
- 计数与分值做归一化

不建议把整套 observation 作为“全离散状态”处理。

建议理解为：

- 动作空间是离散的
- 状态输入是混合特征组成的连续向量

## 16. 归一化方案

第一版先采用简单直接的归一化方式：

- 布尔值：`0/1`
- one-hot：`0/1`
- 其余数值特征：按自然上界缩放到 `0-1`

当前约定：

- 不做复杂标准化
- 不做均值方差归一化
- 第一版不引入需要映射到 `[-1, 1]` 的差值特征作为默认设计

示例：

- `points / 5`
- `cost / 7`
- `missing_after_discount / 7`
- `gems / 10`
- `gem_slots_left / 10`
- `discounts / 7`
- `round_number / 30` 或 `round_number / 40`
- `deck_counts` 按各自等级牌堆最大数量归一化

原则：

- 第一版优先统一、简单、稳定
- 后续如果加入如 `score_gap` 这类可正可负特征，再单独讨论是否映射到 `[-1, 1]`

## 17. 当前已达成的设计共识

目前已明确的结论：

1. 不暴露未翻开的卡牌内容。
2. `reserved_cards` 是公开信息，可以编码。
3. 动作空间采用固定 `45` 维离散动作。
4. observation 采用固定长度向量。
5. 卡牌特征中的 `missing_after_discount` 只考虑折扣，不考虑当前手牌宝石。
6. `gem_slots_left` 保留。
7. `total_gems` 不保留，因为与 `gem_slots_left` 重合。
8. `owned_cards_count` 暂不保留，因为贡献有限。
9. 对手预留卡采用 3 个固定槽位，编码方式与其他卡一致。
10. 第一版环境暂时保留现有规则实现中的后处理行为：
   - 超过 10 宝石时随机弃牌
   - 多个贵族可访问时默认选择第一个
11. 第一版 observation 的归一化原则为：布尔和 one-hot 保持 `0/1`，其余数值特征按自然上界缩放到 `0-1`。

## 19. V2 TODO

以下特征当前版本先不加入 observation，但可作为第二版候选迭代方向：

- `score_gap`
  - 例如 `self_score - opponent_score`
- `can_trigger_last_round`
  - 例如“当前是否存在一步买牌即可触发最后一轮”的布尔特征
- 将当前环境中的后处理决策显式化，并建模为 agent 的显式决策步骤
  - 超过 10 宝石时不再随机弃牌，而是作为 agent 决策
  - 多个贵族可访问时不再默认选第一个，而是作为 agent 决策

## 20. 实现建议

建议后续实现以下接口：

- `encode_observation(game, player_index) -> np.ndarray`
- `get_action_mask(game, player_index) -> np.ndarray`
- `decode_action(action_id, game, player_index) -> Action`

以及后续再封装：

- `gymnasium.Env` 风格环境

## 21. V1 PPO 初始训练设置

第一轮 PPO 训练先采用保守配置，目标是：

- 验证训练链路稳定
- 观察 reward 曲线和 episode 统计是否开始改善
- 快速拿到第一版基线模型

当前第一轮训练设置如下：

- 算法：`sb3_contrib.MaskablePPO`
- policy：`MlpPolicy`
- 动作掩码：启用
- 训练对手：`random`
- seed：`7`
- timesteps：`50000`
- `n_steps = 1024`
- `batch_size = 256`
- `learning_rate = 3e-4`
- `gamma = 0.99`
- `max_episode_steps = 200`
- TensorBoard：启用

当前默认网络结构为：

- 输入维度：`464`
- actor：`464 -> 64 -> 64 -> 45`
- critic：`464 -> 64 -> 64 -> 1`
- 激活函数：`Tanh`

第一轮训练产物应至少包括：

- 训练配置快照
- monitor 统计
- TensorBoard 日志
- 模型文件
- 对随机代理的评估结果

第一轮实际运行信息：

- run name：`v1_ppo_random_seed7_t50k_post_endfix`
- 实际训练时长：约 `3分54秒`
- 训练产物目录：`runs/rl/v1_ppo_random_seed7_t50k_post_endfix`
- 模型文件：`runs/rl/v1_ppo_random_seed7_t50k_post_endfix/model.zip`
- 评估结果：`runs/rl/v1_ppo_random_seed7_t50k_post_endfix/eval_random_20.json`

第一轮实际评估结果（对 `random`，20 局）：

- `wins = 18`
- `losses = 2`
- `draws = 0`
- `invalid_terminations = 0`
- `win_rate = 0.90`
- `avg_reward = 16.9`
- `avg_steps = 31.6`

补充说明：

- 这轮结果是在修复最后一轮结束判定 bug 后重新训练得到的，更可信
- 后续又修复了 late game 的 1-2 色 `take_different_gems` 动作编码 bug，原先统计中的“平局/异常终止”已消失
- 当前 baseline 已经证明 V1 PPO 链路可用，但策略强度还值得继续提高
- 下一步更值得做的是继续细化 reward、提升训练步数、加入更强对手，或扩大评估规模

## 22. 当前状态

本文档当前是“第一版状态空间/动作空间设计稿”。

后续讨论将直接在本文件基础上修订。
