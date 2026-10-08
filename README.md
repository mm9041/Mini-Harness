# mini-harness

一个可运行的轻量级 AI agent harness，用来理解 [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) 的核心架构：插件装配、模型适配、工具执行、会话事件，以及 turn/step 驱动循环。

当前包版本 **0.1.0**。核心使用 Python 标准库，图片读取可选依赖 Pillow。当前实现包含终端和网页界面、15 个默认工具、后台命令作业、子代理、审批、重试及上下文管理，适合学习架构和本地实验。

## 快速开始

需要 Python ≥ 3.10。命令工具需要 PowerShell：优先使用 PATH 中的 `pwsh`，Windows 上可回退到 `powershell`。默认 `workspace-write` 命令沙箱目前只支持 Windows，还需要 Node.js（含 npm）及下述沙箱运行时；其他平台仅安装 PowerShell 不能启用受限命令。文件和网络工具本身不依赖 PowerShell 或 Node.js。

在项目目录运行：

```powershell
cd E:\PycharmProjects\DSH\mini-harness
# Windows 首次使用受限命令前安装沙箱运行时（需要 Node.js/npm 和网络）
python scripts/setup_sandbox.py
python -m mini_harness --mock "确认一下能不能跑通"
```

`--mock` 无需 API Key，不调用真实模型。它请求 `pwsh` 执行固定命令 `echo mini-harness-ok`，再回述工具结果。默认权限为 `workspace-write`，需先安装沙箱运行时；范围内命令自动执行。离线演示：

```powershell
python -m mini_harness --mock --permission workspace-write --no-stream "确认一下能不能跑通"
```

未覆盖步数等默认配置时，正常完成有 2 个 step，命令结果包含 `returncode: 0` 和 `mini-harness-ok`。事件总数随权限初始化、ACL 准备等情况变化，不固定为 11 条。离线模式只替换模型适配器，工具仍会真实执行；它不是自然语言任务模拟器。`--mock --web` 还受已保存的网页服务商配置影响，详见下节。

安装为可编辑包后，可以在其他目录使用 `python -m mini_harness` 或 `mini-harness`：

```powershell
python -m pip install -e .
# 如需 read_image：
python -m pip install -e ".[images]"
```

安装可能需要下载构建依赖，核心 Python 包没有必需的第三方 Python 运行依赖；图片读取和 Windows 命令沙箱另有上述依赖。

## 接入真实模型

默认使用 DashScope / 阿里云百炼的 OpenAI 兼容接口，模型为 `qwen-plus`。在项目目录复制配置模板，再编辑本地 `.env`：

```powershell
Copy-Item .env.example .env
```

已有 `.env` 时直接编辑，避免覆盖现有配置。最小配置如下：

```dotenv
DASHSCOPE_API_KEY=你的密钥
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-plus
```

Git 只提交不含真实密钥的 `.env.example` 模板；`.env` 及其本地变体、`.mini-harness/` 运行数据、`.workbuddy/` 助手记忆、缓存和依赖目录均已加入 `.gitignore`。源码、测试、文档及 `package-lock.json` 应正常提交。忽略规则不会自动移除已经跟踪的文件；如曾误传密钥，应先在对应服务商撤销并重新生成，再清理仓库及相关提交历史。

```powershell
# 单次任务
python -m mini_harness "统计当前目录下有多少个 .py 文件"
# 终端多轮交互；交互终端中也可以省略 --repl
python -m mini_harness --repl
# 网页界面
python -m mini_harness --web --open
```

网页版默认地址为 <http://127.0.0.1:8770>。`--web` 模式下位置参数中的任务不会执行，请在页面输入。

启动配置由命令行参数覆盖环境配置，未配置时使用内置默认值；`.env` 不覆盖已存在的同名环境变量。不同别名同时存在时按下表从左到右取首个非空值，即使高优先级别名来自 `.env` 也会优先使用。自动发现 `.env` 时先查启动工作目录，再查项目根目录；`--env-file PATH` 可指定文件，`MINI_HARNESS_ENV_FILE` 也可指定自动发现的目标。`--cwd` 改变工具工作区，不改变 `.env` 查找位置。本文默认值均指未被配置覆盖的内置值。

| 配置 | 环境变量（同行靠前者优先） | 默认值 |
|---|---|---|
| API Key | `DASHSCOPE_API_KEY`、`DEEPSEEK_API_KEY`、`OPENAI_API_KEY` | 空 |
| 接口地址 | `DASHSCOPE_BASE_URL`、`DEEPSEEK_BASE_URL`、`OPENAI_BASE_URL` | DashScope 兼容地址 |
| 模型 | `DASHSCOPE_MODEL`、`MINI_HARNESS_MODEL`、`OPENAI_MODEL` | qwen-plus |
| Tavily 搜索密钥 | `TAVILY_API_KEY` | 空；配置后优先使用 Tavily |

可用 `--base-url`、`--model` 接入其他 OpenAI 兼容网关。网页设置还提供 DeepSeek、千问、智谱、Kimi 等服务商预设及推理参数映射；具体模型名称、视觉能力和参数支持以服务端为准。

网页保存的服务商配置是独立的运行时设置，不修改 `.env`。Windows 上密钥使用当前用户的 DPAPI 保护，文件访问权限依赖目录 ACL（`chmod(0600)` 不设置 Windows ACL）；其他平台以明文存入权限为 `0600` 的本地文件。公开设置响应只返回是否配置了密钥。

网页启动后会应用 `providers.json` 中保存的活动服务商，其模型和连接信息优先于该次启动的模型参数；没有活动服务商时沿用启动适配器，但仍应用保存的推理强度。此规则也适用于 `--mock --web`：如果已有活动服务商，网页可能调用真实 API，不能仅凭 `--mock` 判断网页处于离线模式。终端单次任务和 REPL 不应用这份网页活动服务商配置。

`qwen3.7-max-preview` 的接口要求开启思考，`none` 会映射为开启并使用 1,024 tokens 思考预算，界面提示实际映射。模型的图像输入能力仍取决于网关和型号；本地能读取图片不代表该模型接口接受多模态输入。

配置内容损坏时，先将原文件原样备份为同目录的 `providers.json.corrupt-<唯一标识>.bak`，再使用空配置，并在厂商设置中显示备份位置。读取或备份失败会报错并保留原文件，不静默重置。备份具有原文件相同的密钥保护形式。DPAPI 密文会随机变化，判断密钥是否复用应比较解密结果。

## 默认工具清单

以实际注册表为准，可随时查看：

```powershell
python -m mini_harness --list-tools
```

默认启用子代理时有 **15 个工具**，`--no-subagents` 移除 `task`。

| 工具 | 主要参数 | 行为 |
|---|---|---|
| `read` | `file_path`, `offset?`, `limit?` | UTF-8 文本分页读取，带行号；offset 从 1 开始，默认 2000 行 |
| `write` | `file_path`, `content` | 原子创建或整体覆盖 UTF-8 文件 |
| `edit` | `file_path`, `old_string`, `new_string`, `replace_all?` | 字面精确替换；默认要求唯一匹配，保留 BOM/CRLF |
| `glob` | `pattern`, `path?`, `limit?` | 按路径模式找文件，包含隐藏文件和 Git 忽略的文件 |
| `grep` | `pattern`, `path?`, `glob?`, `ignore_case?`, `limit?` | 正则检索，返回路径、行号和匹配行；按文件头 NUL 字节启发式跳过二进制文件 |
| `read_image` | `file_path`, `max_edge?` | PNG/JPEG/WebP/GIF；默认最长边 1600，GIF 取首帧；需要 Pillow 和视觉模型 |
| `pwsh` | `command`, `cwd?`, `timeout?`, `background?` | 执行 PowerShell 命令，timeout 单位为秒 |
| `job_list` | 无 | 列出当前会话的作业及状态 |
| `job_output` | `job_id`, `offset?`, `limit?`, `wait?`, `timeout?` | 分段读取输出，offset 按字节计，返回 next_offset；可等待完成 |
| `job_kill` | `job_id` | 终止当前会话的作业及子进程树，不接受任意系统 PID |
| `web_search` | `queries`, `count?`, `topic?`, `time_range?` | 配置密钥后优先 Tavily，网页搜索备用；自动识别新闻，返回来源、时间和诊断；1–5 个查询，每个最多 1–10 条结果 |
| `web_fetch` | `url`, `max_chars?` | 抓取公网 HTTP/HTTPS 文本网页，返回文章链接、页面声明时间及正文截断状态 |
| `ask_user_question` | `question`, `options?` | 提问并等待回答，最多 6 个选项，也可自由输入或跳过 |
| `present` | `files: [{file_path, title?}]` | 将 1–10 个已有文件登记为交付物，提供预览、打开与下载 |
| `task` | `description`, `prompt` | 委派一次性子代理，等待并返回最终结果 |

局部修改优先 `edit`，整体替换使用 `write`。当前默认装配不注册旧的 `shell`、`read_file`、`write_file` 重复入口。

### PowerShell 与后台作业

默认命令入口由 `jobs.py` 提供。每次命令启动独立 PowerShell 进程，不保留上次命令设置的变量或 cd 状态。`cwd` 或命令内部的 `cd` 只影响该次命令，不改变工作区配置。

`background=true` 允许作业跨对话轮次运行。每个独立运行环境的作业服务最多同时运行 8 个作业，父会话及其子代理共享这个限额；网页的不同主会话拥有独立作业服务。输出写入 `jobs/<id>/output.log`，通过 `job_output` 获取。`job_output` 的 `wait=true` 默认最多等 30 秒，最大 60 秒；等待超时不会终止作业。

`pwsh` 和 `job_output` 的 `timeout: null` 均按默认超时处理。为避免 PowerShell 5.1 的进度流 CLIXML 噪声，命令包装默认关闭进度输出。作业服务保留全部运行中作业和最近 50 条已结束作业；更早的记录移除后不能再通过作业 ID 查询，但输出文件仍保留，历史回复中的日志路径仍可使用。完成作业的耗时固定在结束时间。

前台命令受当前轮取消控制；后台作业持续到自然结束、执行超时、`job_kill` 或 harness 退出。Windows 使用 Job Object 管理子进程树。后台作业不跨服务重启恢复，也不会自动发起新的模型对话。

Windows 作业以 `CREATE_SUSPENDED` 创建，先加入 kill-on-close Job，再恢复初始线程，避免分配前派生出未受控的子进程。保留 `Popen`，通过 Win32 Toolhelp 重新打开其关闭的初始线程句柄；无法确认唯一线程、分配或恢复失败时终止启动。嵌套 Job 依赖 Windows 8 及以上支持，具体宿主 Job 限制仍可能导致分配失败。与 DSH 相同，宿主在创建到分配之间被外力终止时仍可能留下挂起进程，不保证原子附加。

`--shell` / `MINI_HARNESS_SHELL` 仍是兼容保留配置，**不会改变默认 pwsh 工具的后端**。`builtin_tools/shell.py` 中的多 shell 实现需要自行装配；默认运行无需 Git Bash。

### 联网、提问与交付

`web_search` 的 `topic` 可选 `auto`（默认）、`general`、`news`。自动模式按查询里的新闻/头条/news/headlines 等词识别新闻；模型也可明确指定。在 `.env` 配置 `TAVILY_API_KEY` 后，普通搜索和新闻搜索均优先调用 Tavily Search API，无需额外安装 SDK。重启后生效。使用 basic 深度，不请求 AI 答案、图片或原始全文；每条查询调用一次，消耗 Tavily 搜索额度。密钥只发送至固定的 Tavily HTTPS API，不进入模型上下文或诊断日志。

Tavily 出错、没有结果或有效结果不足时，自动使用原有备用来源：普通搜索依次使用 Bing 网页、Bing RSS 和 DuckDuckGo HTML；宽泛新闻使用 Google News 头条/地区频道和 Bing News，具体新闻主题优先 Bing News。未配置 Tavily 时保持原有流程。结果按 URL 去重，普通 Tavily 结果保留 API 的相关性顺序，网页备用结果做关键词筛选；新闻仍严格检查发布时间。诊断保留实际 provider 及失败原因，401/403、限流、额度不足等不会被误报为“网上没有相关信息”。不会绕过验证码或付费墙。

`time_range` 可选 `day/week/month/year/any`，新闻默认最近 24 小时，普通搜索默认不限时间。新闻在本地检查 Tavily 的 `published_date` 或 RSS 的 `pubDate`，过滤缺少可解析且带时区的时间、超出范围或明显未来的条目，并返回 `published_at`、`publisher`、`provider`；这些时间来自搜索源声明，未经独立核验，`checked_at` 只是检索时间。普通网页的时间范围交由搜索源过滤，工具明确提示无法在本地确认时效。宽泛新闻的 `search_scope` 描述 API 优先和备用头条频道策略，实际调用来源见 `attempts`。Google News 链接标记 `link_kind=aggregator`，不能冒充已读取的出版方原文。

正常搜索流程为每个查询返回 `status=ok/partial/unavailable`，以及各来源的失败原因、接受数和过滤数 `attempts`；取消或查询级异常可能只返回 `query/error/results`。Tavily 使用搜索源相关性排序，网页备用来源使用基础关键词过滤，两者都不保证语义相关或新闻真实性。一个查询失败不会丢弃其他查询的结果；所有查询都失败时工具标记错误。外部源被拦截或关键词过滤过严仍可能导致结果不足，工具不会宣称互联网没有相关内容。

例如：`{"queries":["国际新闻 今日最新"],"topic":"news","time_range":"day","count":5}`。得到线索后应调用 `web_fetch` 阅读具体文章，核对发布时间、来源和内容。聚合链接打不开时可搜索完整标题；只抓到首页或片段时应明确说明，重要或矛盾事实需独立来源交叉核验。

`web_fetch` 默认返回最多 20,000 正文字符，可配置 1,000–100,000 字符，响应限制 2 MiB；不执行 JavaScript、不登录网站，不解析 PDF/Office 正文。HTML 结果额外返回最多 20 个去重的 HTTP/HTTPS 链接（优先较长标题）、页面明确声明的发布时间/修改时间以及抓取时间；缺失时不从首页日期或抓取时间猜测文章发布时间。链接从完整响应提取，正文截断后仍可用于继续导航；导航识别是启发式，不能保证每条都是文章。初次访问和重定向前会拒绝解析为非公网地址的 URL，但后续 HTTP 连接未固定到已验证 IP，不能视作完整网络隔离。网页内容始终作为未经事实核验的外部资料使用。

`ask_user_question` 与审批是两套机制。即使处于 `danger-full-access`（完全访问）预设，问题仍等待用户回答；跳过、取消或超时都不表示同意。默认等待最多 10 分钟，网页刷新后可以继续回答仍在等待的问题。终端提问采用可取消的异步输入，输入期间超时和取消仍生效；没有交互界面时返回错误，不替用户决定。嵌入调用可通过 `UserQuestions(ctx, timeout=秒数)` 或 `interaction.plugin(question_timeout=秒数)` 调整超时，必须为有限正数。自定义同步 responder 应快速返回，耗时或交互操作应使用异步 responder。回答与取消同时就绪时保留已接受的回答，重复回答不覆盖首个回答。

`present` 不创建文件，也不验证内容是否完成。应先生成并检查文件，再登记交付。网页按本轮登记顺序将交付卡片显示在回复结束之后，成功、失败或停止时均保留已登记文件的入口。下载使用会话登记的文件 ID，不提供任意路径下载接口。文本、代码、Markdown、HTML/SVG 源码按纯文本安全预览，图片和 PDF 使用浏览器预览，Office 等文件可下载后打开。文本预览最多读取 1 MiB，下载保留完整文件。卡片指向磁盘当前文件，并非内容快照；文件移动或删除后访问会报错。

## 网页与终端交互

网页版使用标准库 `ThreadingHTTPServer`、SSE 和单文件 HTML/JavaScript 前端，网页界面无需前端构建或 Node.js 服务；Windows 沙箱命令执行依赖 Node.js，相关依赖由 `package.json` 和 `package-lock.json` 管理。当前支持：

- 流式回答、推理内容、工具轨迹、状态和耗时展示。
- 新建及重开历史会话。删除会移入归档，可在“设置 → 管理归档”恢复或永久删除。
- 归档默认保留 7 天：网页服务启动、每小时以及打开归档列表时清理过期会话；程序关闭期间不执行清理，下次启动补清理。旧 `.trash` 记录按元数据文件的修改时间计算保留期。永久删除不可恢复；清理范围仅为已归档的会话文件，不删除工作区文件。
- 按工作区分组展示历史，点击“工作区”右侧搜索图标打开历史搜索。
- 新建对话时选择工作区；取消选择不改变当前对话，已有对话不能更换目录。
- 服务商、模型、推理强度及权限设置。
- 审批卡片、提问卡片、附件与文件交付。

一个服务实例支持多个对话同时执行。运行时仍可新建对话或点击侧栏切换，原任务继续在后台运行；同一对话不能重复提交重叠轮次。每个对话有独立的模型适配器、工作目录、取消令牌、审批、上下文计量、压缩和作业服务；切换只改变展示，不重新加载正在运行的日志。侧栏标出“运行中”和“等待确认”，停止按钮只影响当前对话；后台任务的审批须切回对应对话处理。正在运行的对话不能删除，其他历史仍可操作。历史列表按最后发送消息的时间展示，浏览或重开不会更新该排序时间。

上下文用量提示显示最近一次主回复的输入缓存命中 token 数与比例，读取 OpenAI `prompt_tokens_details.cached_tokens` 或 DeepSeek `prompt_cache_hit_tokens`。缺失或非法统计显示“服务商未报告”；缓存 token 仍占上下文空间，命中率不等于费用节省比例。原始 usage 随回复保存，刷新后可恢复。

在厂商、模型和思考设置不变时，支持 `reasoning_content` 的思考模式会保留每条历史 assistant 工具消息中已记录的 reasoning；不再根据最后一条 user 的位置或后续工具结果决定是否回放。追加新一轮或摘要指令不会因此改写旧消息。普通正文回复不额外携带 reasoning，厂商和思考模式限制仍生效。保留历史 reasoning 会增加请求输入，是否降低账单需结合服务商报告的缓存用量与实际价格评估；本地前缀一致性测试不等于线上缓存命中保证。

压缩另记 `compaction/start`、`compaction/usage`、`compaction/end`，用操作 ID 关联；只有成功的 `compaction` 事件改变历史投影。摘要被拒绝但已返回 usage 时也保留用量，主请求的 token 校准不受其污染。上下文提示另列最近摘要的输入、输出、缓存和完成状态。只有开始没有结束的记录表示未完成，不推测其费用或自动重放调用。

溢出恢复优先识别适配器提供的错误码，缺少专用码时仅匹配明确涉及模型上下文容量的措辞。`max_tokens` 参数校验、泛化的 token limit 或频率上限不触发压缩；鉴权、额度、限流与服务故障先按各自类别处理。误判可能增加摘要调用、缩短逐字尾部并损失缓存复用，因此不采用宽泛子串判断。缓存计费和服务端前缀复用效果仍以服务商报告为准。

后台输出继续落盘，切回或刷新时恢复已提交消息和当前流式片段。前端请求和事件携带对话 ID，避免迟到的响应、停止或审批落入另一对话。成功完成的回复会把过程折叠到“已完成 + 耗时”（不足一分钟只显示秒，一分钟起显示分秒，一小时起显示小时、分、秒），最终正文和交付文件保留可见；未完成、失败和停止的轮次保留过程。仅在最终正文下方提供复制图标按钮和完成时间（如 `10月4日 20:10`），过程回复不显示模型名称、复制按钮或时间；复制只包含该条回复的 Markdown 正文，不含思考、工具输出、模型名称或时间。时间来自已保存的消息/轮次事件，按浏览器本地时区显示，切换对话或刷新不会改写为当前时间。

并行对话使用独立运行环境，但同一工作区内的文件仍是共享的，不会自动创建独立检出或文件锁。服务退出会停止全部任务；不支持进程退出后继续后台运行。网页仍面向单个本地用户，多浏览器窗口共享当前选中的对话。嵌入方使用自定义插件时可设置 `ui.runtime_factory(source_ctx, cwd)` 构建独立 Context；带不可复制状态的模型适配器需实现 `clone_for_conversation()`，标准 OpenAI 兼容适配器可直接复制。

侧栏底部同一行提供设置和明暗主题切换；连接状态在页面顶部居中显示。当前目录显示在右上角文件夹按钮左侧，该按钮用本地资源管理器打开当前工作区。输入区的上下文圆环可点击查看用量，发送与停止共用一个按钮，分别显示向上箭头和方块；运行时按 Enter 不会触发停止。

设置面板提供 12–24px 的对话字号（默认 16px）和中文 / English 界面语言（默认中文）。字号只影响消息正文、思考内容、工具输出和对话中的代码，不改变输入框、侧栏或设置控件。偏好保存在当前浏览器；语言切换会刷新界面并保留草稿，不翻译已有对话、模型名称、文件路径或用户定义的历史标题。模型厂商配置仍可从设置面板进入。

模型回复语言与界面语言独立。系统提示要求：明确指定的回复/翻译语言或持续语言偏好优先，否则跟随当前用户消息的语言；`hi`、`hello` 等英文问候用英文回应，环境说明、文件路径、工具输出及引用材料不作为回复语言依据。这是提示规则，实际遵循情况取决于模型。修改系统提示规则后需重启服务。

网页权限菜单从服务端读取“只读 / 工作区内修改 / 完全访问”三个预设，同时显示实际沙箱模式与审批策略。

每条消息最多附带 8 个文件，总计 10 MiB，保存到当前工作区 `.mini-harness/uploads/`。聊天正文保留用户输入，附件单独显示文件名和大小；文件路径及读取提示只加入模型输入。符合旧版自动追加格式的上传消息兼容此显示方式，不改写原日志。附件由模型通过文件工具读取；上传 Office 或 PDF 不会自动提取正文。

页面“停止”取消当前轮；网页服务进程中的 Ctrl+C 停止整个服务。终端任务中第一次 Ctrl+C 请求协作式取消，再次按下可强制退出。

| REPL 命令 | 用途 |
|---|---|
| `/help`、`/tools`、`/events` | 帮助、工具清单、当前会话事件统计 |
| `/model [名称]` | 查看可用模型或切换模型 |
| `/permission [预设]` | 查看或切换 read-only / workspace-write / danger-full-access |
| `/access [档位]` | 兼容入口；标准预设下 ask→workspace-write、allow/full→danger-full-access、deny→read-only，推荐用 `/permission` 查看实际预设 |
| `/compact` | 立即尝试压缩较早历史 |
| `/sessions` | 列出会话目录中的最近文件 |
| `/resume [序号或路径]` | 切换历史；省略参数使用最近会话 |
| `/save [路径]` | 保存当前会话 |
| `/exit`、`/quit` | 退出 |

### 网页输入框命令

输入 `/` 会显示命令提示，选择后按发送或 Enter 执行：

| 命令 | 行为 |
|---|---|
| `/compact` | 立即尝试压缩旧历史，显示结果；需要一次摘要模型请求 |
| `/status` | 查看窗口容量来源、下一次输入估算及上次实测 |
| `/tools` | 列出当前工具 |
| `/permission [预设]` | 查看或切换 read-only / workspace-write / danger-full-access |
| `/new` | 打开新建对话和工作区选择弹窗 |
| `/settings` | 打开界面设置，可继续进入模型厂商设置 |
| `/help` | 查看帮助 |

命令不会作为用户消息进入模型历史；其他命令不请求模型。命令不能附带文件。压缩可用停止按钮取消，历史过短或压缩禁用时会提示，不声称成功。手动和自动压缩共用已有摘要与日志机制；原始事件保留，历史恢复后摘要继续生效。自动压缩按可用窗口信息和校准估算判断（默认 80%），容量未知时仍使用标记为估算的窗口；近期完整工具组或工具表本身很大时，不能保证一次压缩便降到阈值以下。

## 会话保存与恢复

会话以 JSONL 保存，每行一个事件。`Session.append()` 向内存序列追加事实，`derive_messages()` 从事件重建模型消息。**追加式事件语义不等于每条事件立即写盘**：保存由 CLI、REPL 或 Web UI 调用，文件采用完整 JSONL 的原子替换，避免保存中断损坏原检查点。Windows 替换时遇到短暂的访问/共享冲突会有限重试；持续失败仍报告错误并保留旧检查点，不保证本轮新增事件已落盘。

```powershell
python -m mini_harness --save-session .mini-harness/sessions/demo.jsonl "介绍当前目录"
python -m mini_harness --resume .mini-harness/sessions/demo.jsonl "继续解释核心模块"
python -m mini_harness --continue "接着上次的任务"
```

单次 CLI 新会话默认不自动保存，使用 `--save-session`；REPL 新会话可用 `/save`。已恢复的会话有来源文件，可写回该文件。Web UI 会保存会话并管理历史索引。

`--continue` 和 REPL `/sessions` 查询会话目录，按文件修改时间排序，不扫描所有工作区 JSONL；这与网页历史按最后消息时间展示不同。

恢复时会跳过无法解析的事件行并报告修复信息，补齐中断留下的未完成工具结果及 step/turn 尾部，保持后续请求的工具协议完整。当前没有会话格式版本迁移链。

| 数据 | 默认位置 |
|---|---|
| 会话目录 | 启动目录下 `.mini-harness/sessions/`，可用 `--session-root` 覆盖 |
| 作业日志 | 启动目录下 `.mini-harness/jobs/`；指定 session root 时使用其父目录下 `jobs/` |
| 网页服务商配置 | 会话目录的父目录下 `providers.json` |
| 上传附件 | 当前工作区下 `.mini-harness/uploads/` |
| 外溢结果 | `~/.mini-harness/spill/`，可用 `--spill-root` 覆盖 |

工作区与会话存储位置独立，`--cwd` 不会自动迁移会话和作业目录。

`--continue` 忽略 0 字节会话文件，网页自动保存也不写出完全没有事件的新会话。会话读取兼容 UTF-8 BOM，新增保存使用 UTF-8、LF 换行。允许有效事件序号有间隔，但必须为递增正整数；重复或倒退序号会拒绝续跑，保留原文件，避免错误解释压缩边界。

读取时跳过坏行会记录修复提示。首次覆盖该源文件前，原始字节备份为同目录的 `<会话文件名>.recovery-<唯一标识>.bak`；备份失败或源文件在读取后改变时不覆盖。另存为新文件保留旧文件。备份含完整原始会话内容，不自动清理。事件观察者异常会记入错误日志并继续通知其他观察者，不撤销已经追加的事件。

REPL 的 `/resume`、`/save` 等命令遇到文件或数据错误时提示后继续，失败切换不会摘掉当前会话观察者。CLI 收尾保存失败会单独提示，不覆盖在途异常或原有非零退出码；执行原本成功但保存失败时退出码为 2，已经生成的回答仍显示。非致命观察者、缓存清理和子代理结束通知错误默认只记录简短诊断；对应 logger 启用 DEBUG 时附带堆栈，库本身不改写宿主的全局 handler 配置。

## 架构与运行流程

### 插件装配

`app.py` 的 `HarnessConfig` 保存配置，`build_plugins()` 构造插件列表，`build_context()` 调用内核挂载。

`kernel.py` 提供服务容器 `Context` 和插件描述 `Plugin`。插件用 `ctx.provide()` 注册服务，用 `inject` 声明依赖；内核在依赖就绪后装载，无法满足依赖时报错。`ctx.effect()` 登记清理函数，`dispose()` 逆序清理资源。

| 事件模式 | 行为 |
|---|---|
| `emit` | 依次等待监听器，忽略返回值，用于通知 |
| `serial` | 依次等待监听器，返回最后一个结果 |
| `waterfall` | 监听器通过 next() 委托下游，也可直接返回拦截 |

`emit` 本身不是并行调度器。模型请求、工具执行和 UI 通过事件接口连接。

### turn 与 step

**turn** 是一次用户输入引发的处理过程；**step** 是一次模型决策及其提出的工具调用处理，失败时请求可能重试。默认装配每轮最多 64 步。

```text
用户输入
  → turn/start、user/message
  → 检查取消及压缩需求
  → 从会话投影历史，组装系统提示和工具 schema
  → 请求模型，记录完整 assistant/message
      ├─ 有工具调用：执行、记录结果 → 下一步
      └─ 无工具调用：返回回答 → turn/end
```

离线演示的主要事件序列（省略权限初始化、ACL 准备等附加事件）：

```text
turn/start
user/message
step/start
assistant/message   请求 pwsh({"command": "echo mini-harness-ok"})
tool/call
tool/result
step/end
step/start
assistant/message   根据工具结果回答
step/end
turn/end            stopped=final
```

每一步重新从事件投影历史，系统提示分区渲染，工具 schema 通过模型协议字段传入。运行时分区读取当前工作区和模型，避免切换后仍使用启动时的配置。

### 工具执行、流式与提前派发

```text
tools/pre-execute → 工具 handler → tools/execute → tools/post-execute
    审批拦截          实际执行         通知             结果处理
```

工具由注册表提供，审批和外溢通过事件挂载。处理器的普通异常转换为 `ToolResult(is_error=True)`，让模型根据错误决定下一步。

流式增量通过 `agent/assistant-stream` 给 UI，日志记录拼装完成的消息。启用提前执行时，仅 `safe_to_prefetch=True` 的工具有资格在流结束前派发，例如文件读取和检索。写入、PowerShell 和子代理不提前派发，避免流失败重试造成重复副作用。提前任务可并发完成，结果按调用顺序提交。

### 审批、取消与重试

权限预设现在组合真正的文件/命令沙箱与审批策略，与 DeepSeek Harness 的标准预设一致：

| 预设 | 文件及命令写入边界 | 审批 |
|---|---|---|
| `read-only` | 限制模型工具和子命令写入；宿主仍会保存会话、附件和作业日志 | `ask`，可为一次调用申请更宽权限 |
| `workspace-write`（默认） | 工作区及该会话私有临时目录可写 | `ask`，范围内调用不逐次询问，扩大权限需要批准 |
| `danger-full-access` | 无沙箱限制 | `never`，不弹审批 |

`never` 的含义仍是拒绝需要审批的请求；完全访问预设下，普通操作已经不需要扩大权限。读取沿用 DSH 的非隔离语义：文件工具可以读取当前用户能读的路径；网络工具仍保留自己的公网 URL 限制。`job_kill` 只能操作当前会话自己的作业。

Windows 使用官方 `@deepseek-ai/dsh-sandbox-windows-acl@0.1.7-rc.2`，由 Node.js runner 创建 WRITE_RESTRICTED/Low-IL 进程和 Job 对象；文件工具单独在实际写入前检查同一模式。依赖用 package-lock.json 固定。首次安装：

```powershell
python scripts/setup_sandbox.py
python -m mini_harness --permission workspace-write --web
```

安装器默认把运行时放在 `~/.mini-harness/sandbox-runtime`，避免位于模型可写工作区内。可用 `MINI_HARNESS_SANDBOX_RUNTIME_ROOT` 指定安装目录。运行时或 Node 位于可写工作区内时，受限命令会拒绝运行。Python 本体仍可独立启动；缺少 Node/runner、后端初始化失败或不支持的平台都返回错误，绝不自动改为无隔离执行。目前只接入 Windows 原生命令沙箱；其他平台仍可使用文件围栏，受限命令会返回 `SANDBOX_UNAVAILABLE`。

启动配置用 `--permission` 或 `MINI_HARNESS_PERMISSION_MODE`。已有会话通过网页菜单或 `/permission read-only|workspace-write|danger-full-access` 切换；`/permission` 查看当前值。预设及其 sandbox/approval 值写入 `permission/preset`，恢复后保留，子代理继承父会话策略。无新预设事件的旧会话中，`deny` 保守迁移到只读，旧 `allow` 不会自动扩大为完全访问。

需要扩大一次操作的范围时，模型提供 `sandbox_permissions`（`workspace-write` 或 `danger-full-access`）以及非空 `justification`。批准只对这次工具调用有效，不改变会话预设；拒绝、取消、无审批人或权限切换均不能产生授权。审批仍记录 `approval/asked` / `approval/decided`。切换不追溯终止已经启动的后台作业，要结束它们请停止或终止作业。

工具元数据 `permission` 表示操作类型，`sandboxed=True` 只允许真正消费并执行 `ToolCallContext.sandbox_mode` 的处理器使用。未分类或没有围栏的写入/执行插件在受限模式下不能运行，只能显式申请单次完全访问。旧 shell 插件没有接入新 runner，不会冒充受限 shell。插件属于可信宿主代码；沙箱不会防御恶意 Python 插件本身。

Windows 后端限制写入与删除，不隔离所有读取、网络或进程可见性。它沿用 DSH 的已知边界：硬链接别名、低完整性标签和工作区 ACL 授权会产生持久目录元数据变化；普通运行结束不会撤销工作区的 standing grant。命令采用隐藏控制台以兼容受限 PowerShell 的 DLL 初始化，后台生命周期仍由 Job 对象管理。

`workspace-write` 在启动命令前检查所选工作区的 ACL，不绑定某个固定目录；文件系统和目录权限必须支持该后端。DSH 会移除部分用户组权限，因此只有组 Modify 权限仍可能不能写；设置并传播 Low 完整性标签还需要 `WRITE_OWNER`。对于当前用户拥有且可修改 DACL 的工作区，harness 会先备份根目录的安全描述符，再仅补缺失的当前用户 SID 可继承 Modify 和 `WRITE_OWNER`，不接管属主、不授予 FullControl、不删除已有拒绝规则；NULL DACL 会明确拒绝自动修改。这些权限按 Windows 继承规则传播，为既有子文件的 Low 标签初始化提供必要权限；受保护的子目录、不同属主或显式拒绝仍可能阻止初始化。受限子进程的能力 SID 不含 `WRITE_DAC/WRITE_OWNER`，因此它不能利用宿主用户新增的权限改写 ACL。已有所需权限的目录不重复改写；`read-only` 和 `danger-full-access` 不触发准备。新建私有临时目录继续单独授予当前用户访问权。

备份保存在沙箱运行时目录的 `acl-backups/*.json`（默认 `~/.mini-harness/sandbox-runtime/acl-backups`），记录工作区路径、用户 SID、修改前的根目录 owner/group/DACL SDDL 和新增权限；会话记录 `permission/workspace-acl-repaired` 事件及备份路径。备份仅用于诊断和人工恢复，没有自动回滚功能，也不包含 DSH 随后设置的 Low 标签或完整子目录安全描述符。备份完整发布后才修改 ACL；进程内锁不防止其他进程同时修改 ACL。备份失败、目录属主不同或无权修改 ACL 时，命令启动前返回包含路径和原因的 `SANDBOX_UNAVAILABLE`，保持当前权限预设，不自动切换完全访问。若看到 `SetNamedSecurityInfoW ... Win32 5 ... grantWrite(...)`，应检查报错目录的 ACL、显式拒绝和文件系统支持，而非反复申请完全访问。

旧的 `MINI_HARNESS_APPROVAL` / `MINI_HARNESS_ALLOW_OUTSIDE` 环境变量只用于嵌入式兼容路径，标准启动以 `MINI_HARNESS_PERMISSION_MODE` 为准。独立 `--approval` / `--allow-outside` CLI 开关已弃用并会提示改用预设，避免显示一种策略却执行另一种。嵌入式调用可显式使用 `HarnessConfig(permission_preset=None)` 保留原审批接缝；这条兼容路径没有命令沙箱，单元测试中的旧组件使用它，新预设另有原生集成测试覆盖。

取消采用令牌和检查点协作。已提交的工具调用必须得到对应结果，包括取消结果，避免恢复后缺失工具响应。

重试策略在驱动器步骤边界执行，默认总尝试次数 3：

- 429、5xx 和部分网络故障按策略退避重试。
- 普通 400、401、403 不重试；上下文超限另行识别，尝试压缩并重建请求。
- 先记录 retry/scheduled，再等待；成功或放弃记录 retry/recovered、retry/gave-up。
- 退避可取消。`--retry-always` 启用持续重试策略，仍受错误分类和取消控制。

### 上下文预算与子代理

| 机制 | 解决的问题 | 实现 |
|---|---|---|
| 外溢 spill.py | 单条结果太长 | 默认超过 12,000 字符时尝试保存完整结果，返回预览和路径 |
| 压缩 compaction.py | 历史整体太长 | 投影裁剪旧工具结果，用额外模型请求生成旧历史摘要 |
| 子代理 subagent.py | 探索过程不必进入父历史 | 独立子会话执行，只交回最终结果 |

压缩参考 `deepseek-harness-master/packages/compaction/compaction-basic` 的策略：

- 默认阈值为 `min(窗口 × 0.8, 窗口 − 输出预留 − headroom)`，检查完整请求的校准估算（正文、图片、工具参数、system 和工具 schema）。阈值属于完整输入预算，system/schema 已在比较的总量中计入，不能再次从阈值扣除。先应用工具结果裁剪，再判断是否仍需摘要；请求前与回答完成后都会检查。
- 默认从末尾累积保留 `(窗口 − 输出预留) × 0.16` tokens，并向前调整边界以保持工具调用与结果配对。`/compact` 和确认溢出后的压缩默认保留最新一个完整消息/工具组，不要求达到自动阈值。
- 摘要请求复用当前 system、工具 schema 和被覆盖的原始消息结构，在末尾追加结构化交接指令；不再把每条消息截成 4000 字符，也不会执行摘要返回的工具调用。八节摘要覆盖用户目标、技术概念、文件代码、错误修复、待办、当前工作、下一步和关键约束，并合并旧检查点。
- 只有完整、非空的正文摘要，且包含检查点包装后的估算 token 数确实小于原历史时，才追加 `compaction` 事件。截断、仅推理内容、工具调用、取消或失败都不会覆盖历史。同一会话不能并发压缩：自动检查遇忙跳过，显式压缩遇忙返回 `CompactionBusyError`（`code=busy`）并显示原因。生成期间新增的尾部消息会保留。
- 上下文溢出最多压缩恢复一次，且必须成功缩小后才重建请求重试；普通网络重试策略不变。原始日志保留，旧 `compaction` 记录仍能恢复。

自动压缩失败会写入 `command/result`，网页和终端展示具体原因；该提示可随日志恢复，但不会进入模型消息。手动 `/compact` 开始时展示实际保留范围。当前参考源码 `compaction-basic/src/index.ts` 的 `compactNow()` 同样向 `selectCompactableRange` 传入 `0`，因此默认只保留最新完整单位是与当前 DSH 一致的明确手动策略。

`MINI_HARNESS_MAX_HISTORY_TOKENS=0` 只关闭独立历史上限，窗口压力检查仍启用；正数上限只统计 messages（包括工具参数和图片），不含 system/schema。`MINI_HARNESS_KEEP_RECENT=0` 使用 token 保留策略，设置正数则显式使用原来的最近 N 条下限策略（含手动压缩）。输出预留和 headroom 默认都是 0，80% 阈值本身留出窗口余量；可按实际模型配置调大。这两个参数仅控制压缩预算，不设置服务商输出上限。mini-harness 仍使用校准估算和简单追加日志，没有移植 DSH 的完整 surface 事务、图片 offload 或按模型路由的配置系统。

外溢文件是有保留期的缓存，不是永久归档；保存失败不保证完整内容可取回。`--no-spill` 只是不装配外溢插件，不为所有工具安装统一截断器，各工具自身的分页和长度限制仍有效。

默认保留 7 天，以会话目录最近新增文件的时间计。启动时清理；后续外溢写入每小时检查一次，网页服务闲置时也每小时检查。清理失败会记录警告。每个会话最多保留 1000 个外溢文件、总计 256 MiB（嵌入调用可传 `max_session_files` / `max_session_bytes`）；达到配额时拒绝保存新全文并明确返回截断提示，不提前删除未过期的旧快照。会话日志通常只有预览，过期清理后全文不保证可恢复。

文件使用排他创建及 `0600` 创建权限，已有同名文件不会被覆盖；Windows 文件访问仍依赖目录 ACL。`tools/post-execute` 支持多个改写者：原地改写后 `return await nxt()`，替换对象后 `return await nxt(call, replacement, context)`，链尾返回最新结果。spill 会继续委托，包括落盘失败后的截断结果；直接返回 `ToolResult` 仍表示有意短路。

`token_meter.py` 使用字符估算和模型返回的 usage 校准。窗口大小依次取显式配置、服务商模型目录返回的原始容量、模型近似表或默认值；服务商容量不取整、不截断。界面注明容量来源。

空闲时圆环显示下一次输入预计占用，包含按已记录压缩投影的历史、已保存回复和工具结果、系统提示及工具 schema；不包含草稿、待上传附件，不预留模型输出空间，也不提前执行自动摘要。运行中显示当前请求输入量，API 返回 prompt_tokens 时采用实测，否则估算。提示单列当前会话、当前模型上次输入实测值；不再把窗口减输入量称为可用剩余空间。字符和图片公式仍有误差，不能当作精确分词器或计费报表。

`task` 是进程内、一次性委派，父代理等待结束。description 用于显示，prompt 必须自包含，因为子代理从空对话开始。子代理有独立事件，共享工具、运行环境和取消机制，审批继承父会话策略。默认最多嵌套一层，使用 ContextVar 跟踪深度。失败、取消或步数耗尽返回错误；成功返回最终结果，长结果还可能受子代理自身长度限制。

已创建的子会话在成功、失败或取消收尾时自动保存到会话目录，包括工作目录、父会话 ID 和结束时的有效权限；返回值与父日志记录实际保存路径。网页普通历史默认隐藏带 `session/parent` 标记的子会话（已有日志同样适用）；在父会话的委派记录中点击“查看子会话日志”，可只读查看原始事件日志，不切换聊天、不恢复执行。日志未保存或已不存在时明确提示。终端仍可通过 `/sessions` 查找日志并显式恢复。落盘成功后释放原运行时的 session/agent 注册引用；保存失败会明确提示并保留内存对象，不宣称可以从磁盘回查。强制结束进程仍可能丢失尚未收尾的子会话。

执行结果与日志持久化分别判断：执行成功但日志保存失败时，`task` 仍返回成功及有界的最终答案，并附“日志未落盘”和具体保存错误，避免因持久化失败重复执行。执行失败、取消、步数耗尽或没有最终答案仍是错误；若同时保存失败，两种原因都会报告。

持久事件为 `subagent/start` / `subagent/end`；深度限制拒绝单独记 `subagent/refused`，不创建子会话。`subagent/started` / `subagent/finished` 是供界面挂接轨迹的瞬时通知，不替代持久日志。结果长度仍可通过 `subagent.plugin(result_max_chars=...)` 配置，尚无单独的 env/CLI 选项。

这里隔离的是对话历史，不是进程或文件系统。当前没有父历史 fork、持续消息式子代理和后台子代理调度；重开已保存日志不恢复原来的父子运行时关联。

## 文件导航

| 文件 | 职责 / dsh 概念对应 |
|---|---|
| [app.py](mini_harness/app.py) | 默认插件装配，类似 base bundle |
| [kernel.py](mini_harness/kernel.py) | Context、依赖、事件与清理，类似 Cordis 内核 |
| [agent_loop.py](mini_harness/agent_loop.py) | Agent 注册表和 turn/step 驱动 |
| [llm.py](mini_harness/llm.py) | 请求、结果、schema、流事件与适配器服务 |
| [adapters/openai_compat.py](mini_harness/adapters/openai_compat.py) | HTTP、SSE、工具调用片段拼接 |
| [adapters/mock.py](mini_harness/adapters/mock.py) | 离线演示与脚本化测试适配器 |
| [session.py](mini_harness/session.py)、[user_messages.py](mini_harness/user_messages.py) | 事件、消息投影、附件正文分离、恢复与原子保存 |
| [system_prompt.py](mini_harness/system_prompt.py) | 系统提示分区 |
| [tools.py](mini_harness/tools.py) | 工具注册和执行流水线 |
| [builtin_tools/files.py](mini_harness/builtin_tools/files.py) | 当前文件及图片工具，复用 fs.py 路径检查 |
| [jobs.py](mini_harness/jobs.py)、[process_group.py](mini_harness/process_group.py) | PowerShell 作业与进程树管理 |
| [web_tools.py](mini_harness/web_tools.py)、[interaction.py](mini_harness/interaction.py) | 联网、提问、交付 |
| [web_search.py](mini_harness/web_search.py)、[tavily.py](mini_harness/tavily.py) | 搜索来源调度、结果筛选和 Tavily API |
| [approval.py](mini_harness/approval.py)、[interrupt.py](mini_harness/interrupt.py)、[retry.py](mini_harness/retry.py) | 审批、取消、重试 |
| [permissions.py](mini_harness/permissions.py)、[windows_acl.py](mini_harness/windows_acl.py) | 权限预设、命令沙箱接线和工作区 ACL 准备 |
| [spill.py](mini_harness/spill.py)、[compaction.py](mini_harness/compaction.py)、[token_meter.py](mini_harness/token_meter.py) | 上下文及用量管理 |
| [subagent.py](mini_harness/subagent.py) | 子代理委派 |
| [cli.py](mini_harness/cli.py)、[repl.py](mini_harness/repl.py)、[console.py](mini_harness/console.py) | 终端入口、交互与轨迹 |
| [webui.py](mini_harness/webui.py)、[webui_sessions.py](mini_harness/webui_sessions.py)、[webui_static/index.html](mini_harness/webui_static/index.html) | HTTP/SSE 服务、多会话管理和网页前端 |
| [history.py](mini_harness/history.py)、[providers.py](mini_harness/providers.py) | 历史索引及恢复、服务商与推理设置 |
| [envfile.py](mini_harness/envfile.py)、[directory_picker.py](mini_harness/directory_picker.py) | 环境文件与目录选择 |

建议阅读顺序：app.py → kernel.py → agent_loop.py → session.py → tools.py → adapters/openai_compat.py，再看审批、重试和上下文管理。builtin_tools/shell.py 和 builtin_tools/fs.py 中还保留旧工具插件，不代表默认装配启用它们。

## 扩展方式

### 增加工具

插件依赖 tools 服务，注册名称、描述、JSON Schema 和处理器：

```python
from mini_harness.kernel import Context, Plugin
from mini_harness.tools import Tool, ToolResult


def hello_plugin():
    def apply(ctx: Context):
        def hello(args, context):
            return ToolResult(f"你好，{args['name']}")

        ctx.effect(ctx.tools.register(Tool(
            name="hello",
            description="返回一句问候。",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
            handler=hello,
            permission="read",  # 此示例只生成文本，没有文件写入或命令执行
        )))

    return Plugin(name="hello", apply=apply, inject=("tools",))
```

把 hello_plugin() 加入 app.build_plugins() 即可。safe_to_prefetch 默认关闭，只有重复执行不修改外部状态的工具才应考虑开启。新增工具应声明 `permission`；未声明时默认 `unknown`，按受控操作处理。不要仅为避免审批而把有副作用的工具标成 `read`。

### 更换适配器与拦截执行

适配器实现 `generate(request) -> GenerateResult`，需要原生流式时再实现 `stream(request)`，通过 `ctx.llm.register_adapter()` 注册和 `ctx.llm.use()` 选择。只有 generate 的适配器也可由服务层拆分完整结果为流事件。

内置兼容适配器使用标准库 HTTP/SSE，无需 OpenAI SDK。可对照 [examples/openai_sdk_adapter.py](examples/openai_sdk_adapter.py) 学习 SDK 实现，运行该示例需要额外安装相应 SDK。工具参数拼装和安全提前派发等策略仍需保留。

执行拦截通过 waterfall：

```python
from mini_harness.kernel import MODE_WATERFALL
from mini_harness.tools import ToolResult


async def block_write(call, context, nxt):
    if call.name in ("write", "edit"):
        return ToolResult("此插件禁止文件修改", is_error=True)
    return await nxt()


ctx.on("tools/pre-execute", block_write, mode=MODE_WATERFALL)
```

事件模式属于接口契约，调用并等待 nxt() 才会继续下游。

## 命令行与环境变量

完整参数可用 `python -m mini_harness --help` 查看。

| 参数 | 行为 / 默认值 |
|---|---|
| `[任务]`、`--repl` | 单次任务；交互终端不传任务进入 REPL；非 TTY 显式使用 --repl |
| `--web`、`--host`、`--port`、`--open` | 网页服务；127.0.0.1:8770，端口 0 随机分配 |
| `--cwd PATH` | 工具工作区，默认启动目录 |
| `--model`、`--base-url`、`--api-key` | 模型连接配置 |
| `--timeout SEC`、`--max-steps N` | 请求超时 120 秒，每轮最多 64 个模型决策步 |
| `--no-stream`、`--no-early-tools` | 关闭流式或安全工具提前执行 |
| `--permission read-only\|workspace-write\|danger-full-access` | 沙箱与审批权限预设 |
| `--no-spill`、`--spill-root PATH` | 关闭外溢或改变目录 |
| `--no-compaction`、`--history-budget N` | 关闭压缩或设置额外历史上限（默认 0，按窗口策略） |
| `--context-window N` | 覆盖上下文窗口大小 |
| `--retry N`、`--retry-always`、`--retry-delay SEC` | 总尝试次数 3，持续重试策略，退避基数 0.5 秒；N=0 不安装重试插件 |
| `--no-subagents`、`--subagent-depth N` | 移除 task 或改变默认嵌套深度 1 |
| `--shell-timeout SEC` | 命令默认超时 60 秒 |
| `--shell NAME` | 兼容保留，不影响默认 pwsh 后端 |
| `--session-root PATH`、`--save-session PATH` | 会话目录与显式保存位置 |
| `--resume PATH`、`--continue` | 恢复指定或会话目录中最近的会话 |
| `--persona TEXT` | 覆盖人格提示 |
| `--env-file PATH` | 指定环境文件 |
| `--dump-events` | 未选择保存路径时输出会话 JSONL |
| `--list-tools`、`--mock`、`--quiet`、`--version` | 工具清单、离线演示、隐藏事件轨迹、版本 |

下表环境变量均以 `MINI_HARNESS_` 为前缀：

| 名称后缀 | 默认值 |
|---|---|
| `MAX_STEPS`、`TIMEOUT`、`SHELL_TIMEOUT` | 64、120、60 |
| `STREAMING`、`EARLY_TOOLS` | 1、1 |
| `PERMISSION_MODE` | workspace-write；可选 read-only / danger-full-access |
| `SANDBOX_RUNTIME_ROOT` | ~/.mini-harness/sandbox-runtime |
| `SPILL`、`SPILL_ROOT`、`SPILL_INLINE_CHARS` | 1、默认外溢目录、12000 |
| `COMPACTION`、`MAX_HISTORY_TOKENS`、`KEEP_RECENT` | 1、0、0 |
| `COMPACTION_THRESHOLD_RATIO`、`COMPACTION_RETAIN_RATIO` | 0.8、0.16 |
| `COMPACTION_HEADROOM_TOKENS`、`COMPACTION_RESERVED_COMPLETION_TOKENS` | 0、0（仅压缩预算预留） |
| `CONTEXT_WINDOW` | 0；优先用服务商报告容量，否则按模型近似表或兜底值估算 |
| `RETRY_ATTEMPTS`、`RETRY_ALWAYS`、`RETRY_BASE_DELAY` | 3、0、0.5 |
| `SUBAGENTS`、`SUBAGENT_DEPTH` | 1、1 |
| `WEB_HOST`、`WEB_PORT` | 127.0.0.1、8770 |
| `ENV_FILE` | 自动发现；指向不存在的路径可禁用自动加载 |
| `SHELL` | 兼容保留，不改变默认 pwsh 工具 |
| `PERSONA` | 配置类支持；当前 CLI 会用 --persona 或内置人格覆盖该值 |

`.env.example` 可作变量模板，审批说明与当前操作类型策略一致；当前装配以 app.py、jobs.py 为准。

## 验证

```powershell
python -m unittest discover -s tests -t .
```

测试使用脚本化模型和本地测试服务，不需要真实模型或 Tavily API Key。覆盖插件依赖、恢复、工具边界、流式重试、取消、审批、外溢、压缩、子代理、后台作业、网页历史和设置、附件、提问及交付；前端逻辑测试需要 Node.js，Windows 原生沙箱测试还需要 PowerShell 和沙箱运行时，图片测试需要 Pillow。部分环境条件不满足时会跳过对应测试，应同时检查测试报告中的 skipped 数量。

测试通过不代表所有真实网关都可用，工具调用、推理字段和视觉能力仍需针对目标服务验证。

## 当前边界与后续方向

- 内核只有 emit、serial、waterfall，没有完整热重载或声明式 profile/bundle/patch 装配。
- 会话为裸 JSONL，没有版本化迁移；保存检查点与内存事件追加不是同一时刻。
- 提前工具派发没有统一并发上限；作业服务的 8 个运行上限是另一层限制。
- 审批没有永久允许某条命令的记忆；Windows 使用 DSH 原生写入沙箱，读取/网络不隔离，其他平台命令沙箱尚未接入。
- token 与窗口有估算成分，压缩没有完整旧图片 offload 机制。
- 子代理只有一次性、父等待子完成的模式，没有持续消息、父历史 fork 或外部代理后端。
- 网页支持本地多会话并行执行，没有多用户认证或统一全局并发上限。进程退出后任务不会自动继续执行，但可重新加载已保存历史并手动续聊。
- 已有后台命令作业、历史索引、新建对话时选择工作区、附件和本地服务商配置；尚无完整 MCP、技能加载、跨重启任务调度和独立工作区实体体系。

后续可先补工具并发上限、审批记忆和会话迁移，再扩展技能渐进加载、可续聊子代理与更精确的上下文计量。

每个 step 是一次模型决策及其工具调用批次，不是一次完整用户任务；工具结果交回模型后还需要下一步，才能继续调用或给出最终回答。默认每轮 64 步，`--max-steps N` 或 `MINI_HARNESS_MAX_STEPS=N` 可调整，必须是正整数。触及上限会保留已完成的工具结果并显示未完成状态；在同一会话发送“继续”会从现有历史继续，新一轮重新计步。修改环境变量后需重启服务，既有会话可从历史记录恢复。
