# high-agent

`high-agent` 是独立的 Python `3.13t` free-threaded 并行 agent runtime。它把模型 tool call、验证任务和动态任务统一 lower 成 `AgentTaskSpec`，由 run-scoped `CausalRuntime` 按依赖和资源冲突并行调度；模型控制面使用多泵 planner，在工具任务运行期间继续并发规划下一批动作。

`hermes-agent/` 只是本地参考代码，不属于本项目提交范围。

## 环境

必须使用 noGIL/free-threaded CPython：

```bash
.venv/bin/python -c "import sysconfig; print(sysconfig.get_config_var('Py_GIL_DISABLED'))"
```

输出应为 `1`。安装/刷新依赖：

```bash
/home/yhsim/.local/bin/uv pip install --python .venv/bin/python -e .
```

## 首次配置

交互式设置：

```bash
PYTHONPATH=src .venv/bin/python -m high_agent setup
```

非交互环境可以直接用变量：

```bash
export HIGH_AGENT_PROVIDER=custom
export HIGH_AGENT_MODEL=gpt-5.4
export HIGH_AGENT_API_KEY=...
export HIGH_AGENT_BASE_URL=https://example.com/v1
export HIGH_AGENT_API_MODE=chat_completions
export HIGH_AGENT_MODEL_TIMEOUT_SECONDS=900
export HIGH_AGENT_MAX_PLANNER_REQUESTS=2
export HIGH_AGENT_MAX_WORKERS=8
export HIGH_AGENT_MAX_ITERATIONS=200
export HIGH_AGENT_DELIVERY_DEBOUNCE_SECONDS=0.05
```

默认配置目录：

- `~/.config/high-agent/config.yaml`
- `~/.config/high-agent/secrets.yaml`

可用 `HIGH_AGENT_HOME` 指向其他目录。解析优先级是 `CLI > HIGH_AGENT_* > config.yaml > 默认值`。

## 配置文件

`~/.config/high-agent/config.yaml` 支持以下参数：

```yaml
model:
  timeout_seconds: 600        # 模型 HTTP 超时（秒）

runtime:
  max_workers: 8              # 工具执行并行线程数
  max_planner_requests: 4     # planner 最大并发模型请求数
  delivery_debounce_seconds: 0.05  # delivery 合并窗口（秒）
  critical_path_fanout: 2     # 关键路径触发阈值（唤醒 N 个 waiting task 时触发 refill）
  critical_path_signal_budget: 16  # 单次 run 内关键路径信号上限

agent:
  max_iterations: 200         # 单次 run 最大 planner 请求数
  tool_use_enforcement: auto  # 工具使用强制策略（auto/true/false）
```

所有参数都可通过 CLI 参数或环境变量覆盖：

| 参数 | CLI | 环境变量 | config.yaml |
| --- | --- | --- | --- |
| 最大迭代数 | `--max-iterations` | `HIGH_AGENT_MAX_ITERATIONS` | `agent.max_iterations` |
| 并行线程数 | `--max-workers` | `HIGH_AGENT_MAX_WORKERS` | `runtime.max_workers` |
| planner 并发 | — | `HIGH_AGENT_MAX_PLANNER_REQUESTS` | `runtime.max_planner_requests` |
| 模型超时 | `--model-timeout` | `HIGH_AGENT_MODEL_TIMEOUT_SECONDS` | `model.timeout_seconds` |
| delivery 合并 | `--delivery-debounce` | `HIGH_AGENT_DELIVERY_DEBOUNCE_SECONDS` | `runtime.delivery_debounce_seconds` |
| 关键路径阈值 | — | — | `runtime.critical_path_fanout` |
| 关键路径预算 | — | — | `runtime.critical_path_signal_budget` |

## 使用

本地 runtime demo：

```bash
PYTHONPATH=src .venv/bin/python -m high_agent runtime-demo
```

一次性任务：

```bash
PYTHONPATH=src .venv/bin/python -m high_agent run "在当前目录创建 demo/a.txt，内容为 hello"
PYTHONPATH=src .venv/bin/python -m high_agent run --trace --yes "运行测试并修复失败"
PYTHONPATH=src .venv/bin/python -m high_agent run --model-timeout 900 "构建一个完整项目"
PYTHONPATH=src .venv/bin/python -m high_agent run --delivery-debounce 0.02 "执行多个短工具任务"
```

交互会话：

```bash
PYTHONPATH=src .venv/bin/python -m high_agent chat
PYTHONPATH=src .venv/bin/python -m high_agent chat --trace
```

TTY TUI 采用 Hermes-style 非全屏底部输入区：输出以纯文本写入真实终端 scrollback，终端原生滚动可用。终端宽度足够时，底部输入区右侧会显示 runtime 并行任务动态图；面板读取真实 `RuntimeLedger`，用 `READY ---> RUNNING ---> DONE` 的 lane 图展示 `AgentTask` 状态、耗时、等待依赖和失败摘要，不显示模型自述的 todo。输入 `/` 会弹出 slash command 菜单；长文本不会逐字符触发命令补全，大段粘贴会先折叠成本地 paste 引用，提交时再完整展开给 agent。Enter 提交当前输入但不退出界面；Ctrl-J 插入换行；Ctrl-D 退出。
普通本地项目文件操作不会弹审批；workspace 及其项目父目录会被视为可信项目范围。当工具请求高风险审批时，TUI 会在终端输出区显示审批请求；直接输入 `y`、`s`、`a` 或 `n` 进行一次、本会话、永久允许或拒绝。也可以先用 `/yes` 开启自动允许危险操作。

常用 slash commands：

```text
/help
/model
/tools list
/tools enable mcp
/tools disable terminal
/history [n]
/sessions
/resume <session_id-prefix>
/compress [budget]
/memory
/plugins
/status
/usage
/tasks
/panel on
/panel off
/logs
/trace on
/trace off
/workspace [path]
/yes
/no
/run <prompt>
/exit
```

默认 `terminal`、`run_python`、`run_tests`、MCP 调用和保护路径会走审批。明确允许危险操作时使用：

```bash
PYTHONPATH=src .venv/bin/python -m high_agent run --yes "运行测试并修复失败"
```

模型 HTTP 默认超时为 600 秒。大项目构建如果出现 `model request timed out`，可使用 `--model-timeout`、`HIGH_AGENT_MODEL_TIMEOUT_SECONDS`，或在 `config.yaml` 中设置 `model.timeout_seconds`。模型请求失败时会自动重试 5 次（每次间隔 10 秒）；planner 请求超过 120 秒未返回会被自动取消并重新发起。

runtime 并行度由工具的 `ResourceAccess` 决定。v0.4.20 起，HTTP fetch 这类 external read 不再全局互斥；`terminal`、`run_python`、`run_tests` 默认仍按 unknown workspace mutation 串行化，但模型或用户明确提供 `mutates_workspace=false` 和可选 `reads`/`writes` hint 时，可以与无冲突任务并行。

v0.5 起，`AgentRunController` 使用 ledger 变化驱动的 planner：每个 run 默认最多允许 2 个不同状态下的模型规划请求同时 in-flight，但同一 ledger/context 触发只启动一个 planner。任一工具 delivery 或 status/gate 事实到达后，会基于最新 ledger snapshot 立即补发 planner；最终文本只作为候选，等 runtime idle 且没有 in-flight planner 后再通过 `CompletionGate`。模型请求仍使用现有 `model_client.complete(messages, tools=...)` 和 provider 原生 tool-call 协议。公开 `delegate_task` 工具和 `delegate` toolset 已移除；需要并行工具时继续使用普通 tool call、`multi_tool_use.parallel` 或批量工具。

## Provider 支持

v0.4.13 起真实模型请求默认全部使用流式 SSE；测试 fake client 或不支持 `stream()` 的注入 client 会保留 JSON fallback。v0.4.0 支持 API-key/custom endpoint 路径，内置三类 transport：

- `chat_completions`
- `anthropic_messages`
- `codex_responses`

模型 tool call 会先进入 `ToolCallNormalizer`，支持裸工具名、`functions.*` 命名空间和 Codex/Hermes 风格 `multi_tool_use.parallel` wrapper。`write_many_files` 会在 runtime 内 lower 成多个 `write_file` 子任务，让 runtime 能按文件资源并行调度；但回灌模型时会聚合成原始 tool_call id 的一个结果，避免 Chat Completions 兼容网关报 `tool call and result not match`。流式 delta 会在 transport 内合并成 provider-neutral `NormalizedResponse` 后再进入 agent loop。工具结果按 provider 原生格式回灌：Chat Completions 使用 `role=tool`，Anthropic 使用 `tool_result` block，Responses 使用 `function_call_output`。

v0.4.18 起模型请求前会执行 Hermes-style tool protocol sanitizer：修复 malformed tool-call arguments，移除 orphan tool results，为缺失结果的 tool call 补 stub，并按 assistant tool_call 顺序重排 provider-facing `role=tool` 结果。这个 sanitizer 只作用于发给 provider 的 API copy，不改 runtime ledger。v0.4.19 起同一套参数修复也会应用到模型刚返回、尚未 lower 到 runtime 的当前 tool call，避免坏 JSON 在执行前直接变成 `invalid tool arguments`。

Provider catalog 包含：`openai`、`openrouter`、`anthropic`、`xai`、`deepseek`、`alibaba`、`zai`、`kimi`、`kimi-coding`、`minimax`、`lmstudio`、`ollama`、`custom`。

OAuth、ACP、Bedrock SDK 暂不直接复制 Hermes 实现；CLI 会识别并提示使用兼容 endpoint 或后续 adapter。

## v0.4.0 能力

- Hermes-like prompt_toolkit 交互：TTY 底部输入区、终端 scrollback 输出、history、状态栏、slash commands、JSONL transcript。
- `/usage` 和状态栏显示上下文估算、模型调用次数和 token usage；provider 不返回 usage 时使用本地估算兜底。状态栏的 `turn=` 是从用户发送请求到整轮 agent 完成的耗时，`task=` 是 runtime 内部工具/任务累计耗时。右侧 runtime 图形面板和 `/tasks` 显示真实 ledger snapshot、runtime wall time、累计 task time、running task elapsed、等待边和最近完成/失败任务；`/panel on|off` 可切换右侧面板；`/logs` 显示 transcript、trace 和 tool result store；`/trace on|off` 控制之后 run 的 runtime trace。
- SQLite session persistence：`/sessions`、`/resume`。
- Toolsets：`file`、`edit`、`search`、`terminal`、`todo`、`code`、`browser-lite`、`mcp`。
- 项目构建工具：`list_tree`、`read_many_files`、`write_many_files`、`patch_file`、`run_tests`、search/todo/Python code execution/http fetch/MCP fake bridge。
- `AgentRunController` 由 runtime delivery 和 planner futures 共同驱动模型继续工作；planner 使用短 transcript、ledger digest 和 durable context，不把每个异步 planner 的完整中间 transcript 追加进总上下文。
- planner 不会在同一 ledger/context 触发下一次性打满并发槽；重复 tool call 仍会按 `(snapshot_seq, tool_name, canonical_args)` 去重。trace 会记录 `planner.started`、`planner.completed`、`planner.ignored_duplicate`、`planner.final_candidate`、`planner.accepted_final`。
- `CompletionGate` 防止 pending task、目录空壳构建、失败测试和未说明的 blocked/failed 状态被提前 final。
- `PromptPolicy` 只注入通用 tool-use enforcement / execution discipline，不做项目构建专用 prompt 增强。
- 大工具输出进入 `ToolResultStore`，模型只拿摘要和 `result_id`。
- `ApprovalManager` 支持 `ask|auto|deny` 和 `once|session|always`。
- `ContextCompressor` 和 `MemoryStore` 支持上下文压缩与长期事实 digest。
- `PluginManager` 支持 `HIGH_AGENT_HOME/plugins/*/plugin.yaml` 注册工具、命令和 hooks。

## 测试

```bash
PYTHONPATH=src uv run --with pytest pytest -q
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m compileall -q src tests
PYTHONPATH=src .venv/bin/python -m high_agent runtime-demo
PYTHONPATH=src .venv/bin/python -m high_agent setup --non-interactive
printf '/usage\n/tasks\n/logs\n/exit\n' | PYTHONPATH=src .venv/bin/python -m high_agent chat
```

真实模型 smoke 可用已配置模型短测：

```bash
PYTHONPATH=src .venv/bin/python -m high_agent run --max-iterations 3 "请只回复 OK，不要调用工具。"
```

## 项目结构

- `src/high_agent/runtime/`：调度器、组件账本、资源冲突、trace。
- `src/high_agent/agent/`：`MainAgent`、多泵 `AgentRunController`、总上下文和 prompt/tool-call 策略。
- `src/high_agent/llm/`：provider resolver、model client、transport adapters。
- `src/high_agent/tools/`：带 `ResourceAccess` 的工具 registry、toolsets、核心工具、大输出存储。
- `src/high_agent/memory/`：session DB、压缩、长期记忆。
- `src/high_agent/plugins/`：plugin manifest 加载与注册。
- `src/high_agent/cli/`：setup/run/chat/runtime-demo 和 slash commands。
- `spec/`：中文 spec 文档金字塔。
- `tests/`：验收测试。
