# high-agent Benchmark

面向并行 agent runtime 的评测框架，支持 `high_agent` 与 `hermes-agent` 对比评估。

## 评测维度

### 类别 A：并行调度效率 (Parallel Scheduling Efficiency)

测试 runtime 在已知依赖图上的并行调度能力。任务具有明确的 DAG 结构，
评估调度器能否识别无冲突任务并最大化并行度。

**指标：**
- `speedup_ratio`：实际墙钟时间 / 串行基线时间（越小越好）
- `parallelism_utilization`：实际并行度 / 理论最优并行度（越接近 1 越好）
- `scheduling_overhead`：调度开销占总时间比例
- `conflict_detection_accuracy`：资源冲突判断正确率

### 类别 B：多步工具使用 (Multi-Step Tool Use)

测试 agent 在多步骤任务中正确选择和调用工具的能力。
包含文件操作、搜索、终端命令等组合场景。

**指标：**
- `task_success_rate`：任务最终完成率（二值）
- `step_accuracy`：每步工具调用正确率
- `tool_selection_f1`：工具选择的 precision/recall
- `token_efficiency`：完成任务消耗的 token 数
- `step_efficiency`：实际步数 / 最优步数

### 类别 C：复杂规划与协调 (Complex Planning & Coordination)

测试 planner 在复杂目标下的分解、委派和协调能力。
包含需要多轮规划、子任务发现、错误恢复的场景。

**指标：**
- `plan_quality`：规划分解的合理性（人工/LLM-as-judge 评分）
- `delegation_efficiency`：子任务委派的并行利用率
- `error_recovery_rate`：遇到失败后成功恢复的比例
- `completion_time`：端到端完成时间
- `context_utilization`：上下文窗口利用效率

## 综合指标

- `overall_score`：加权综合分 = 0.3×A + 0.4×B + 0.3×C
- `cost_efficiency`：score / total_tokens（性价比）

## 使用方法

```bash
# 运行全部评测
PYTHONPATH=src python -m benchmark.runner --agent high_agent

# 运行单个类别
PYTHONPATH=src python -m benchmark.runner --agent high_agent --category parallel

# 对比两个 agent
PYTHONPATH=src python -m benchmark.runner --agent high_agent --compare hermes

# 查看结果
PYTHONPATH=src python -m benchmark.report
```

## 目录结构

```
benchmark/
├── README.md
├── __init__.py
├── runner.py              # 评测主入口
├── report.py              # 结果报告生成
├── config.py              # 评测配置
├── adapters/              # agent 适配器
│   ├── __init__.py
│   ├── base.py            # 抽象接口
│   ├── high_agent.py      # high_agent 适配
│   └── hermes_agent.py    # hermes-agent 适配
├── evaluators/            # 指标计算
│   ├── __init__.py
│   ├── parallel.py        # 类别 A 评估器
│   ├── tool_use.py        # 类别 B 评估器
│   └── planning.py        # 类别 C 评估器
├── tasks/                 # 任务数据集
│   ├── parallel.jsonl     # 并行调度任务
│   ├── tool_use.jsonl     # 多步工具使用任务
│   └── planning.jsonl     # 复杂规划任务
└── workspaces/            # 任务执行沙箱模板
```
