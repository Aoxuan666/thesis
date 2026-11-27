# 交接材料：鸡舍巡检数字孪生系统（PoultryTwin）

> 本目录是给**其他智能体 / 其他开发者**阅读的交接材料，不参与论文正文。
> 建议阅读入口：**[`AGENT-HANDOFF.md`](./AGENT-HANDOFF.md)** —— 一页纸说清项目定位、该读哪些文件、哪些坑已经踩过。

## 目录

| 路径 | 内容 | 用途 |
|---|---|---|
| [`AGENT-HANDOFF.md`](./AGENT-HANDOFF.md) | 交接简报 | 全景定位、5 条已知坑、提示词模板 |
| [`01-架构/1.3.1-开发文档.md`](./01-架构/1.3.1-开发文档.md) | 开发文档 | 框架图、技术栈、8 个模块、6 个关键算法、API、安全、术语表 |
| [`01-架构/1.3.3-部署文档.md`](./01-架构/1.3.3-部署文档.md) | 部署文档 | 环境依赖、启动顺序、Nginx、验证与排错 |
| [`01-架构/1.3.6-数据库设计说明.md`](./01-架构/1.3.6-数据库设计说明.md) | 数据库设计 | 6 张表结构、索引策略、实测字段差异 |
| [`01-架构/schema.sql`](./01-架构/schema.sql) | 建表脚本 | utf8mb4、索引、约束 |
| [`02-索引/1.3.4-源代码文件清单.csv`](./02-索引/1.3.4-源代码文件清单.csv) | 文件索引 | 164 个源文件的路径 / 语言 / 模块 / 行数 |
| `03-核心源码/` | 8 个核心脚本 + 4 个运行时配置 | 信息密度最高的数据链路与场景生成代码 |

## 系统一句话

Unity 2021.3.44f1c1 构建 WebGL 包嵌入 Web 管理平台，按栋舍参数程序化生成笼架与笼位；云端下发巡检指令后，孪生端以 `afterId` 增量拉取机器人上报的异常记录，在三维场景中落位标注并联动信息面板。后端为 4 个 Java 开放接口，HMAC-SHA256 七段签名鉴权。

## ⚠️ 脱敏说明（重要）

本目录为**公开可见**的副本，已做以下替换，**不能直接编译运行**，需要填回真实值：

| 占位符 | 原内容 | 出现位置 |
|---|---|---|
| `{APP_KEY}` / `{APP_SECRET}` | 开放接口应用标识与密钥 | `03-核心源码/` 下 `RobotOpenApiClient.cs`、`AbnormalChickenApiClient.cs` |
| `{host}:{port}` | 后端服务地址 | 各 `*ApiClient.cs`、`Resources/*.json`、部署文档 |
| `{robot-ip}:1448` | 机器人底盘默认地址 | `RobotOpenApiClient.cs` |
| `{ROBOT_IMEI_A/B/C}` | 三台机器人设备号 | 源码注释与实测记录 |
| `000000` | 原租户 ID | `schema.sql` 默认值、配置与文档示例 |

此外，`02-索引/1.3.4-源代码文件清单.csv` 的字节数是脱敏前的统计，替换后实际字节数会略小，**行数不受影响**。

## 已核实的已知问题（交接必读）

1. 接口文档写 `exceptionPosX/Y`、`exceptionJudgmentValue` 为 decimal，**实测返回字符串**；`exceptionColNo` 实测 14/14 为 `null`、`exceptionRowNo` 为 `-2`。`JsonUtility` 遇 `int` 字段的 `null` 会整批丢数据，故有 `NormalizeRecordJson` 字符串级改写。
2. 场景坐标两套基准：`TryConvertDisplayToInternal` 要求 `row>=1 && cage>=1`（display 1-based → internal 0-based 需 `+1`）；异常自带 `mapX/mapY` 恒为 (11.630, 5.800)，与 `RobotMapLocator` 标定范围不在同一坐标系，不可直接兜底落位。
3. 限流硬约束：同一 appKey 全平台 120 次/分钟，现取值巡检 3 s + 状态 1 s（80 次/分钟），带 `×2` 退避与成功后半衰收回。
4. `/open/monitor/points/latest` 当前返回 `body.code=404`（后端未部署），且 Unity 侧 `FixedMonitoringPointApiClient` 尚未实现签名，真接会 401。

## 不在本目录的内容

完整 Unity 工程（含约 1.5 GB 美术资源）、WebGL 编译产物（约 90 MB）、后端 Maven 工程均在本地工程目录，按需另行提供。
