# 任务 4：Flat vs Hierarchical 公平组织实验

## 实验结论

在 18 条 dev Task、3 次重复、共 108 个真实 DeepSeek V4-Pro 观测中，Hierarchical 明显压缩了上下文并改善了 Skill 选择过程，但没有保持任务完成率。因此，本批数据对主假设给出的结论是：**“节省上下文”成立，“保持或提高 Task Success”不成立。**

| Organizer | Task Success | Selection F1 | Sequence Accuracy | Avg. Context Tokens | Total Tokens | Redundant Call Rate |
|---|---:|---:|---:|---:|---:|---:|
| Flat | 83.33% | 89.81% | 83.33% | 1,104.94 | 51,771 | 10.53% |
| Hierarchical（detail_top_k=3） | 72.22% | 98.15% | 94.44% | 793.06 | 41,942 | 0.00% |
| Hierarchical - Flat | -11.11 pp | +8.33 pp | +11.11 pp | -28.23% | -18.99% | -10.53 pp |

三轮的 Task Success、Selection F1 和 Sequence Accuracy 完全一致，轮间样本标准差为 0；总 Token 受模型输出长度轻微影响。实验中没有 API 失败。

## 公平性与可归因性审计

- 主实验只使用 `BM25Retriever(text_level=brief)`；没有加入 Embedding、Hybrid 或重排。
- 每个 Flat/Hierarchical 配对使用同一条 Task 预先固定的 `candidate_skill_ids`、候选顺序和 `context_budget_tokens`。
- 模型固定为 `deepseek-v4-pro`，temperature=0，thinking disabled，reflection disabled。
- 每个观测创建全新的隔离 `TaskEnvironment`。SQLite 临时文件路径随机，但 setup 后的查询结果和业务状态相同。
- 108 条记录均满足 `skill_context_tokens <= context_budget_tokens`，实际调用 Skill 均属于 `exposed_skill_ids`。
- 预算裁剪只移除完整 Skill 块，不从 Schema 或描述字符串中间截断。

## 为什么出现“选择更好、完成率反而更低”

Hierarchical 用全部 brief 帮助模型识别正确 Skill，因此 Selection F1 和顺序准确率更高；但它只详细披露 BM25 前 3 名。正确 Skill 位于更低排名时，模型可能知道“该调用谁”，却不知道准确参数名。当前 Planner 一次性同时做 Skill 选择和参数生成，没有在选中低排名 Skill 后再请求其 Schema，所以形成了“选对但执行失败”。

分层结果支持这一解释：

- gold rank=5：Flat Task Success 75%，Hierarchical 50%；但 Hierarchical Selection F1/Sequence 均为 100%，Flat 分别为 62.5%/50%。
- candidate_count=20：Flat Task Success 60%，Hierarchical 40%；Hierarchical 平均上下文 886.4，Flat 为 1,130.0 Token。
- budget=800：Flat Task Success 83.33%，Hierarchical 66.67%；Hierarchical 的 Selection F1/Sequence 均为 100%，说明主要瓶颈已从“选哪个 Skill”转为“如何正确传参”。

## 逐任务案例

### 1. `tcalc06`：Hierarchical 选对但缺 Schema

- 候选数 20，gold `aggregate_number_list` 位于第 3 名，预算 800。
- Flat 暴露前三个完整 Skill，生成 `values=[5,7,11,19], operation=sum`，verifier 得到 42，成功。
- Hierarchical 暴露全部 brief，但只详细披露第 1 名；它选对 `aggregate_number_list`，却生成不存在的参数 `list`，缺少必需参数 `values`，执行失败。

### 2. `tdb03`：更低排名 Skill 的参数别名错误

- 候选数 10，gold `update_sqlite_rows` 位于第 5 名，预算 2000。
- Flat 看到完整 Schema，使用 `changes={active: 1}`，数据库状态验收通过。
- Hierarchical 选对同一 Skill，但因未看到详情而使用常见 SQL 表述 `set={active: 1}`，缺少 `changes`，数据库保持原状态，失败。

### 3. `tjson01`：两种披露方式的不同失败机制

- 候选数 20，gold `extract_json_fields` 位于第 5 名，预算 800。
- Flat 只能容纳前 4 个完整 Skill，gold 未暴露，模型误选 `insert_sqlite_row`。
- Hierarchical 以 brief 暴露了 gold，Skill 选择正确；但参数 Schema 未详细披露，模型使用 `json_data` 而不是 `data`，仍执行失败。
- 该案例说明 brief 解决“发现 Skill”，但不能单独解决“准确调用 Skill”。

### 4. `tfile03`：共同失败揭示任务/参数契约问题

- 两种组织器都选对并成功调用 `append_text_file`，但都传入 `content="done"`。
- handler 按字面追加，最终内容是 `startdone`；verifier 要求 `start\ndone`，因此 Task Success 为 0。
- 这是“调用有效、过程选择正确、任务未完成”的标准反例，也说明“追加一行”需要在 Skill 契约或 Planner 提示中明确换行语义，不能归因于 Flat/Hierarchical。

另一个共同失败 `tjson07` 来自把 `$task`（整条自然语言指令）传给要求 JSON 对象的 `data` 参数。它同样属于 Planner 参数绑定限制，而不是组织方式差异。

## Graph 诊断边界

GraphOrganizer 只在 10 条独立依赖诊断任务上做结构检查：9 条没有可排序依赖的逆序；缺失依赖和依赖环各触发 1 次预期警告。当前方法不会自动补召回缺失依赖，因此这些数字只证明拓扑排序和告警工作正常，不代表完整 Graph 方法的任务成功率。

## 下一步解释性改进

最直接的后续消融不是更换检索器，而是实现真正的两阶段披露：先用全部 brief 选 Skill，再只拉取已选 Skill 的详细 Schema 生成参数。这样可以检验 Hierarchical 的成功率损失是否确由一次性 `detail_top_k=3` 导致。共同失败则应通过结构化 Task 输入和明确的文件换行契约单独修复，不能混入组织策略结论。
