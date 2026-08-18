# skill-agent

面向大规模 Skill 库的检索、组织、调用与评测框架。

## 运行流程

```text
Task
  → 多层检索（粗略描述召回 + 详细描述重排）
  → Skill 组织（扁平 / 分层披露 / 依赖图）
  → 结构化规划（技能、参数、顺序）
  → 注册函数执行
  → 反思与任务级评测
```

## 快速开始

```powershell
cd D:\skill\skill-agent
pip install -r requirements.txt
```

打开项目根目录的 `.env`，填写 DeepSeek API Key：

```env
DEEPSEEK_API_KEY=你的_API_Key
```

运行默认实验：

```powershell
python main.py
```

默认配置位于 `configs/default.yaml`。主入口会读取其中的 LLM、检索器、组织器、Top-K 和 Agent 参数。
如需运行 Embedding 或双塔训练，额外安装：

```powershell
pip install -r requirements-ml.txt
```

### DeepSeek V4

默认模型为 `deepseek-v4-pro`，并关闭思考模式，便于稳定比较不同 Skill 检索和组织策略：

```yaml
llm:
  provider: deepseek
  model: deepseek-v4-pro
  temperature: 0.0
  thinking: disabled
  reasoning_effort: high
```

低成本批量实验可改为 `deepseek-v4-flash`。启用思考模式时：

```yaml
thinking: enabled
reasoning_effort: high  # high | max
```

思考模式下不会发送 `temperature`；`reasoning_effort` 只在思考模式下发送。

项目只支持两种运行模式：

```python
LLM(provider="deepseek")  # 真实实验
LLM(provider="mock")      # 离线测试
```

`openai` 包仅作为 DeepSeek 官方 OpenAI 兼容协议的客户端使用，项目不提供 OpenAI 或 Anthropic 模型入口。

## Skill 数据结构

Skill 同时提供粗粒度和细粒度信息：

| 字段 | 用途 |
|---|---|
| `id` | 稳定标识与评测标签 |
| `name` | 规划和函数注册使用的名称 |
| `brief_description` | 第一阶段召回与候选概览 |
| `detailed_description` | 重排、参数选择和详细披露 |
| `category` | 层次分组 |
| `parameters` | 输入 JSON Schema |
| `returns` | 返回值 Schema |
| `tags` | 检索补充词 |
| `examples` | 用法与语义补充 |
| `dependencies` | 前置 Skill ID |
| `metadata` | 风险、权限等扩展属性 |

旧版数据中的 `description` 和 `metadata.depends_on` 仍可由 loader 转换。

## 执行函数注册

数据文件只描述 Skill，不保存可执行源码。执行函数必须由宿主程序显式注册：

```python
def calculator(expression: str) -> str:
    return str(your_safe_calculator(expression))

agent = Agent(
    llm=llm,
    organizer=organizer,
    skill_handlers={"calculator": calculator},
)
```

未注册函数、未知技能、缺少必填参数或执行异常都会记为失败。项目不再通过 `exec()` 运行数据中的任意代码。

## 端到端指标

| 类型 | 指标 | 计算口径 |
|---|---|---|
| 准确性 | **Skill Selection F1** | 每个任务对“已调用 Skill 集合”和标注集合计算 F1，再宏平均 |
| 准确性 | **Sequence Accuracy** | 已调用 Skill ID 序列与标注序列完全一致的任务比例 |
| 最终效果 | **Task Success Rate** | 通过 ground truth、Agent 结果或外部 Judge 判断成功的任务比例 |
| 效率 | **Avg. Skill Calls** | 每个任务的平均 Skill 调用次数 |
| 效率 | **Redundant Call Rate** | 超出标注序列所需次数的调用数 / 总调用数 |
| 上下文 | **Skill Context Tokens** | 每个任务 Skill 上下文的平均估算 Token 数 |
| 成本 | **Total Tokens** | 一次评测中 LLM 的 Prompt 与 Completion Token 总和 |
| 效率 | **Avg. Execution Steps** | 每个任务平均“规划步骤 + Skill 执行步骤”数 |

`Skill Context Tokens` 使用本地中英文估算器，便于比较组织策略；`Total Tokens` 优先采用 API 返回的真实 usage。Mock 模式使用同一估算器。

任务有标准答案时，默认成功判定要求 Agent 执行成功且答案与 `ground_truth` 完全一致。复杂任务可通过 `run_benchmark(..., success_evaluator=...)` 注入环境验证器或独立 Judge。

## 实验

```powershell
python experiments/run_experiment.py
```

脚本比较 BM25、Embedding、多层检索与三种组织策略。默认使用 DeepSeek，完整消融会产生多次 API 调用。离线测试请显式传入 `LLM(provider="mock")`。

## 测试

```powershell
python tests/test_smoke.py
python tests/test_task_metrics.py
python tests/test_agent_execution.py
python tests/test_deepseek_config.py
```

测试使用 Mock LLM，不调用外部 API。
