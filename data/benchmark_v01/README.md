# benchmark_v01 数据说明

## 1. 只回答一个主问题

本数据集不比较检索器，只检验以下假设：

> 在候选列表、候选顺序和上下文预算相同的条件下，当候选 Skill 的名称与 brief 描述相近、适用边界主要位于 detailed 描述中时，Hierarchical 的“全部 brief + 少量 detailed”能否比 Flat 更节省上下文，并保持或提高任务完成率。

主实验固定 brief-BM25、模型、温度、任务初始状态和候选列表。Embedding、Hybrid、两阶段召回与 Graph 不进入该主对比。

## 2. 文件

| 文件 | 内容 |
|---|---|
| `skills.jsonl` | 主实验 24 个 Skill，四个任务族各 6 个 |
| `tasks.jsonl` | 主实验 34 条 Task，dev 18 条、test 16 条 |
| `environment_fixtures.json` | SQLite 可重复初始化脚本；文件初始状态直接保存在 Task metadata 中 |
| `graph_skills.jsonl` | Graph 诊断 Skill 12 个 |
| `graph_tasks.jsonl` | 独立 Graph 诊断任务 10 条 |
| `build_dataset.py` | 确定性重建全部数据与候选切片 |

生成数据：

```powershell
python data/benchmark_v01/build_dataset.py
```

验证数据：

```powershell
python tests/test_benchmark_dataset.py
```

## 3. Hard negatives 怎样形成

四个主任务族分别为：

- `calculation`：整数表达式、小数表达式、百分比变化、方程、单位换算、列表聚合。
- `json_text`：字段提取、键重命名、数组筛选、对象合并、数组排序、CSV 转 JSON。
- `file_operation`：仅创建、仅覆盖、仅追加、复制并保留源、移动、局部修改 JSON 文件。
- `sqlite`：普通查询、插入、更新、删除、聚合查询、多表连接。

同族 Skill 使用刻意相近的 brief 描述；真正决定适用范围的信息写在 `detailed_description`、参数 Schema、返回 Schema及正反例中。负例是功能边界不同的真实 Skill，不是随机改名或无关 Skill。

## 4. 候选数、正确 Skill 排名和预算怎样控制

每条主任务的 `metadata.slice` 保存：

```json
{
  "candidate_count": 10,
  "primary_gold_skill_id": "sjson_rename",
  "target_gold_rank": 3,
  "context_budget_tokens": 1200,
  "raw_bm25_primary_rank": 2
}
```

候选生成过程固定为：

1. 使用当前 `BM25Retriever(text_level="brief")` 对 24 个 Skill 生成完整原始排名。
2. 多步任务先保证所有 gold Skill 都在候选中。
3. 保持 BM25 hard negatives 的相对顺序。
4. 将第一个 gold Skill 移至预设的第 1、3 或 5 位。
5. 截取 5、10 或 20 个候选，并把最终顺序写入 `metadata.candidate_skill_ids`。

这种 rank-controlled fixture 是刻意的实验干预：原始 BM25 排名保存在 `raw_bm25_primary_rank`，而组织方式实验使用完全相同的 `candidate_skill_ids`。这样可以单独观察正确 Skill 位于低排名时，Flat 与 Hierarchical 的差异，不把检索波动混入组织策略结论。

数据覆盖全部 27 个组合：

```text
candidate_count ∈ {5, 10, 20}
target_gold_rank ∈ {1, 3, 5}
context_budget_tokens ∈ {800, 1200, 2000}
```

## 5. Task 标注与验收

每条 Task 都包含：

- `expected_skills`：正确 Skill 集合。
- `expected_skill_sequence`：严格调用顺序。
- `metadata.family`：任务族。
- `metadata.split`：`dev` 或 `test`。
- `metadata.template_id` 与 `variant_id`：表述模板及其变体。
- `metadata.step_count`：一、二或三步任务。
- `metadata.candidate_skill_ids`：组织方式实验必须共用的候选顺序。
- `metadata.environment_fixture`：文件初始快照或 SQLite fixture ID。
- `evaluation`：不向 Planner 暴露的确定性验收规则。

四个任务族分别使用 `exact_match`、`json_match`、`file_state` 和 `sqlite_state`。所有主任务的 `ground_truth` 均为 `null`，用于验证“没有单一文本标准答案，但有明确输出约束或最终状态”时仍可确定性评分。

共有 4 组表述变体；同一个 `template_id` 的全部变体只位于一个 split，避免开发集和测试集之间出现模板泄漏。

## 6. Graph 诊断集

Graph 数据与 Flat/Hierarchical 主实验完全分开，只回答依赖信息是否有用。10 条任务覆盖：

- 两步线性依赖；
- 分支汇合；
- 四至九步长链；
- 严格顺序；
- 指定中间里程碑停止；
- 缺失依赖；
- 依赖环。

缺失依赖和环分别使用隔离候选集，不让两种结构错误互相污染。当前 GraphOrganizer 仍只排序已召回 Skill，不自动补依赖，因此该集合是诊断集，不用于宣称已经实现完整图检索。

## 7. 当前边界

- `run_benchmark(..., use_task_candidate_fixtures=True, use_task_context_budget=True)` 已消费固定候选与预算；组织器按完整 Skill 块裁剪并返回披露证据。
- 文件与 SQLite 初始状态已由 `TaskEnvironment` setup/teardown，24 个 Skill 已接入真实受控 handler。
- 当前数据首先保证可解释的实验压力，不代表真实世界任务分布。
- dev 上的 3 次重复组织实验已完成；test split 继续保留到方案冻结后运行，不用于调 Skill 描述、提示词或预算策略。
