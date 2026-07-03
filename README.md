# BYE-RAG

> 企业级 RAG 系统 + MCP 协议集成 | Production-Ready

基于 **混合检索（Dense + Sparse）** + **RRF 融合** + **MCP 协议** + **完整评估体系**的生产级 RAG 解决方案。

## ✨ 核心亮点

- 🔍 **混合检索**: Dense (语义) + Sparse (BM25) + RRF 融合
- 🛠️ **MCP 集成**: 6 个工具函数，原生支持 Claude Desktop
- 📊 **可视化面板**: Streamlit Dashboard，6 个管理页面
- 🎯 **评估系统**: Ragas 集成，Golden Test Set，回归测试
- 🔄 **链路追踪**: Ingestion + Query 全流程可观测
- 🔌 **可插拔架构**: 支持多种 LLM、Embedding、向量库

---

## 📋 目录

- [快速开始](#-快速开始)
- [依赖说明](#-依赖说明)
- [配置指南](#-配置指南)
- [使用方法](#-使用方法)
- [系统架构](#-系统架构)
- [核心特性](#-核心特性)
- [API 文档](#-api-文档)
- [测试指南](#-测试指南)
- [常见问题](#-常见问题)
- [开发指南](#-开发指南)

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/your-org/BYE-RAG.git
cd BYE-RAG
```

### 2. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装基础依赖
pip install -r requirements.txt
```

### 3. 配置 API Key

编辑 `config/settings.yaml`：

```yaml
llm:
  provider: dashscope
  api_key: YOUR_DASHSCOPE_API_KEY  # 替换为你的 API Key
  model: qwen-max

embedding:
  provider: dashscope  
  api_key: YOUR_DASHSCOPE_API_KEY  # 替换为你的 API Key
  model: text-embedding-v3

milvus:
  uri: ./data/db/milvus.db  # 本地文件模式，无需额外安装
  collection_name: rag_collection
```

### 4. 启动 Dashboard（推荐）

```bash
python scripts/run_dashboard.py
```

访问 http://localhost:8501，开始使用可视化界面：
- 📥 上传和摄取文档
- 🔍 实时查询测试
- 📊 查看追踪数据
- 📝 运行评估

### 5. 或使用命令行

#### 摄取文档

```bash
# 摄取单个文件
python scripts/ingest.py data/documents/report.pdf

# 摄取整个目录
python scripts/ingest.py data/documents/
```

#### 查询

```bash
# 基础查询
python scripts/query.py "什么是机器学习？"

# 混合检索 + 重排序
python scripts/query.py "ML 是什么？" --rerank --top-k 10
```

---

## 📦 依赖说明

### 必需依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| **Python** | ≥3.10 | 运行环境 |
| **pymilvus** | ≥2.3.0 | 向量数据库客户端 |
| **dashscope** | latest | 通义千问 API |
| **langchain** | ≥0.1.0 | 文本分块和工具 |
| **streamlit** | ≥1.30.0 | Dashboard 框架 |
| **pandas** | latest | 数据处理 |

### 可选依赖

| 依赖 | 用途 | 安装命令 |
|------|------|----------|
| **ragas** | RAG 评估系统 | `pip install ragas` |
| **plotly** | Dashboard 图表 | `pip install plotly` |
| **rank_bm25** | BM25 稀疏编码 | 已包含在 requirements.txt |

### 文档加载器依赖

| 依赖 | 支持格式 | 安装命令 |
|------|----------|----------|
| **PyMuPDF** | PDF | 已包含 |
| **python-magic** | 文件类型检测 | 已包含 |

### 安装所有依赖

```bash
# 基础 + 可选
pip install -r requirements.txt
pip install ragas plotly

# 仅安装必需依赖
pip install pymilvus dashscope langchain streamlit pandas PyMuPDF rank_bm25
```

---

## ⚙️ 配置指南

### 完整配置示例

`config/settings.yaml`：

```yaml
# LLM 配置
llm:
  provider: dashscope      # dashscope / openai / azure / ollama
  api_key: YOUR_API_KEY
  model: qwen-max
  temperature: 0.7
  max_tokens: 2000

# Embedding 配置
embedding:
  provider: dashscope
  api_key: YOUR_API_KEY  
  model: text-embedding-v3
  dimension: 2048         # 向量维度

# Milvus 向量库配置
milvus:
  # 本地文件模式（推荐用于开发）
  uri: ./data/db/milvus.db
  
  # 或远程服务器模式
  # uri: http://localhost:19530
  # token: YOUR_TOKEN  # 可选
  
  collection_name: rag_collection
  metric_type: COSINE    # COSINE / L2 / IP

# BM25 稀疏编码配置
sparse_encoder:
  encoder_type: bm25
  k1: 1.5
  b: 0.75

# 混合检索配置
hybrid_search:
  rrf_k: 60              # RRF 融合参数
  alpha: 0.5             # Dense vs Sparse 权重

# 重排序配置
reranker:
  enabled: true
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
  top_n: 5

# 评估配置
evaluation:
  ragas_enabled: true
  threshold: 0.05        # 回归测试阈值（5%）
```

### 环境变量配置

也可以通过环境变量配置（优先级高于 YAML）：

```bash
# LLM
export LLM_PROVIDER=dashscope
export LLM_API_KEY=your_key
export LLM_MODEL=qwen-max

# Embedding
export EMBEDDING_PROVIDER=dashscope
export EMBEDDING_API_KEY=your_key
export EMBEDDING_MODEL=text-embedding-v3

# Milvus
export MILVUS_URI=./data/db/milvus.db
export MILVUS_COLLECTION=rag_collection
```

### 多提供商支持

#### 使用 OpenAI

```yaml
llm:
  provider: openai
  api_key: YOUR_OPENAI_API_KEY
  model: gpt-4

embedding:
  provider: openai
  api_key: YOUR_OPENAI_API_KEY
  model: text-embedding-3-large
```

#### 使用 Ollama（本地）

```yaml
llm:
  provider: ollama
  base_url: http://localhost:11434
  model: llama2

embedding:
  provider: ollama
  base_url: http://localhost:11434
  model: nomic-embed-text
```

---

## 📖 使用方法

### Dashboard 使用（推荐）

#### 1. 系统总览
- 查看系统状态
- 快速统计信息
- 配置概览

#### 2. 数据浏览器
- 浏览已摄取的文档
- 查看数据块内容
- 快速搜索

#### 3. Ingestion 管理
- 上传文档（支持 PDF, MD, TXT）
- 批量摄取
- 查看摄取进度
- 配置 Transform 选项

#### 4. Query 追踪
- 输入查询测试
- 查看检索结果
- 分析性能指标
- 实时追踪数据

#### 5. Ingestion 追踪
- 查看摄取历史
- 分析耗时趋势
- 步骤详情

#### 6. 评估面板
- 加载 Golden Test Set
- 运行 Ragas 评估
- 查看评估历史
- 回归测试

### 命令行使用

#### 数据摄取

```bash
# 基础摄取
python scripts/ingest.py document.pdf

# 带选项
python scripts/ingest.py documents/ \
  --batch-size 50 \
  --enable-transform \
  --enable-trace

# 查看帮助
python scripts/ingest.py --help
```

#### 查询

```bash
# 基础查询
python scripts/query.py "查询文本"

# 混合检索
python scripts/query.py "查询文本" \
  --top-k 10 \
  --rerank \
  --enable-trace

# 查看帮助
python scripts/query.py --help
```

#### MCP Server

```bash
# 启动 MCP Server
python scripts/run_mcp_server.py

# 配置到 Claude Desktop
# 编辑 claude_desktop_config.json:
{
  "mcpServers": {
    "rag-system": {
      "command": "python",
      "args": ["D:\\path\\to\\scripts\\run_mcp_server.py"],
      "cwd": "D:\\path\\to\\RAG-MCP-SERVER"
    }
  }
}
```

### Python API 使用

#### 数据摄取

```python
from src.libs.loader import ComponentLoader
from src.ingestion.pipeline import IngestionPipeline
from src.config.settings import Settings

# 初始化
config = Settings()
loader = ComponentLoader(config)

llm = loader.get_llm()
embedding = loader.get_embedding()
vector_store = loader.get_vector_store()

# 创建 Pipeline
pipeline = IngestionPipeline(
    llm=llm,
    embedding=embedding,
    vector_store=vector_store,
    enable_transform=True,
    enable_trace=True
)

# 摄取文档
result = pipeline.process_file("document.pdf")
print(f"摄取完成: {result['chunks_count']} 个数据块")
```

#### 混合检索

```python
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.query_processor import QueryProcessor

# 初始化
processor = QueryProcessor(llm, embedding, sparse_encoder)
hybrid_search = HybridSearch(
    dense_retriever=dense_retriever,
    sparse_retriever=sparse_retriever,
    enable_trace=True
)

# 处理查询
query_result = processor.process("机器学习的应用")

# 混合检索
results = hybrid_search.search(
    dense_vector=query_result["dense_vector"],
    sparse_vector=query_result["sparse_vector"],
    top_k=10,
    rrf_k=60
)

# 结果
for doc in results:
    print(f"分数: {doc['score']:.3f}")
    print(f"内容: {doc['text'][:100]}...")
```

#### 评估

```python
from src.evaluation import EvalRunner, CompositeEvaluator, RagasEvaluator

# 初始化
ragas_eval = RagasEvaluator(llm=llm, embedding=embedding)
composite = CompositeEvaluator(ragas_evaluator=ragas_eval)
runner = EvalRunner(composite_evaluator=composite)

# 加载测试集
runner.load_test_set("data/test/golden_test_set.json")

# 运行评估
results = runner.run_evaluation(
    include_retrieval=True,
    include_generation=True
)

# 查看结果
summary = results["summary"]
print(f"检索 F1: {summary['retrieval']['avg_f1']:.3f}")
print(f"生成 Faithfulness: {summary['generation']['avg_faithfulness']:.3f}")
```

---

## 📊 系统架构

```
┌─────────────────────────────────────────┐
│    Claude Desktop (MCP Client)          │
└─────────────────┬───────────────────────┘
                  │ MCP Protocol
          ┌───────▼────────┐
          │  MCP Server    │  6 个工具函数
          │  (Tools API)   │
          └───────┬────────┘
                  │
    ┌─────────────┴──────────────┐
    │                            │
┌───▼────────┐          ┌────────▼────┐
│ Ingestion  │          │  Retrieval  │
│  Pipeline  │          │   Pipeline  │
│            │          │             │
│  Load      │          │  Query      │
│  Split     │          │  Dense      │
│  Transform │          │  Sparse     │
│  Encode    │          │  RRF Fusion │
│  Upsert    │          │  Rerank     │
└────┬───────┘          └─────┬───────┘
     │                        │
     │    ┌──────────────┐    │
     └────►  TraceContext ◄────┘
          │  (链路追踪)  │
          └──────┬───────┘
                 │
     ┌───────────▼────────────┐
     │  Milvus Vector Store   │
     │  Dense + Sparse Index  │
     └────────────────────────┘
```

---

## 🔥 核心特性

### 1. 混合检索 (Hybrid Search)

**为什么需要混合检索？**
- Dense 检索擅长理解语义，但对关键词不敏感
- Sparse 检索擅长精确匹配，但不理解语义
- 混合检索结合两者优势，提供更好的检索效果

**工作原理**：
```
用户查询
  ↓
QueryProcessor (查询改写)
  ↓
┌────────────┬────────────┐
│ Dense      │ Sparse     │
│ Retriever  │ Retriever  │
│ (Embedding)│ (BM25)     │
└─────┬──────┴──────┬─────┘
      │             │
      └──────┬──────┘
             ↓
       RRF Fusion (融合)
             ↓
       Reranker (重排序)
             ↓
         最终结果
```

### 2. MCP Server

**6 个工具函数**：

| 工具 | 功能 | 参数 |
|------|------|------|
| `ingest_document` | 摄取文档 | file_path, enable_transform |
| `query` | 混合检索查询 | query_text, top_k, enable_rerank |
| `list_documents` | 列出文档 | - |
| `query_knowledge_hub` | 知识库查询 | collection_name, query |
| `list_collections` | 向量库统计 | - |
| `get_document_summary` | 文档摘要 | doc_id |

**Claude Desktop 集成**：

1. 配置 `claude_desktop_config.json`
2. 重启 Claude Desktop
3. 在对话中直接调用工具

示例：
```
用户: 请帮我摄取 report.pdf 文档
Claude: [自动调用 ingest_document 工具]

用户: 查询"机器学习的应用"
Claude: [自动调用 query 工具并返回结果]
```

### 3. Transform 层

可选的数据增强处理：

| Transform | 功能 | 何时使用 |
|-----------|------|----------|
| **ChunkRefiner** | 优化分块边界 | 提升分块质量 |
| **MetadataEnricher** | 丰富元数据 | 需要更多上下文 |
| **ImageCaptioner** | 图片描述生成 | 处理包含图片的文档 |

启用方式：
```python
pipeline = IngestionPipeline(
    llm=llm,
    embedding=embedding,
    vector_store=vector_store,
    enable_transform=True  # 启用 Transform
)
```

### 4. 链路追踪

**自动追踪**：
- 每个操作步骤
- 耗时统计（ms）
- 错误信息
- 中间结果

**查看追踪**：
```python
from src.trace.trace_context import get_trace_recorder

recorder = get_trace_recorder()

# 获取所有追踪
traces = recorder.get_traces()

# 按类型过滤
query_traces = recorder.get_traces(trace_type="query")
ingestion_traces = recorder.get_traces(trace_type="ingestion")

# 查看详情
for trace in query_traces:
    print(f"查询: {trace['query']}")
    print(f"耗时: {trace['duration_ms']} ms")
    print(f"步骤: {len(trace['steps'])}")
```

### 5. 评估系统

**Ragas 评估指标**：
- **Faithfulness**: 答案忠实于上下文
- **Answer Relevancy**: 答案与问题相关性
- **Context Precision**: 上下文精确度
- **Context Recall**: 上下文召回率

**Golden Test Set**：
- 预定义的测试查询集合
- 用于回归测试
- 跟踪性能变化

**使用示例**：
```python
# 创建测试集模板
runner.create_test_set_template("data/test/template.json")

# 加载测试集
runner.load_test_set("data/test/golden_test_set.json")

# 运行评估
results = runner.run_evaluation()

# 保存结果
runner.save_results("data/eval_results/eval_20260703.json")
```

---

## 🧪 测试指南

### 运行测试

```bash
# 所有单元测试
pytest tests/unit/ -v

# 特定模块
pytest tests/unit/evaluation/ -v
pytest tests/unit/retrieval/ -v

# E2E 测试
python tests/e2e/test_mcp_client.py
python tests/e2e/test_dashboard.py
python tests/e2e/test_e2e_acceptance.py

# 测试覆盖率
pytest --cov=src tests/unit/
```

### 验证脚本

```bash
# 完整功能验证
python scripts/test_all.py

# 接口一致性检查
python scripts/check_interface_consistency.py
```

---

## ❓ 常见问题

### Q1: 如何切换到 OpenAI？

编辑 `config/settings.yaml`:
```yaml
llm:
  provider: openai
  api_key: YOUR_OPENAI_KEY
  model: gpt-4

embedding:
  provider: openai
  api_key: YOUR_OPENAI_KEY
  model: text-embedding-3-large
```

### Q2: 本地运行需要什么？

**必需**：
- Python 3.10+
- 2GB 磁盘空间（向量数据库）
- 4GB RAM

**可选**：
- GPU（加速 Embedding，非必需）

### Q3: 如何使用远程 Milvus？

```yaml
milvus:
  uri: http://your-milvus-server:19530
  token: YOUR_TOKEN  # 如果需要
  collection_name: rag_collection
```

### Q4: Dashboard 启动失败？

检查：
1. 是否安装 streamlit: `pip install streamlit`
2. 端口是否被占用: `lsof -i :8501`
3. 查看错误日志

### Q5: API Key 配置不生效？

检查优先级：
1. 环境变量（最高）
2. config/settings.yaml
3. 默认值

确保环境变量名称正确：
```bash
export LLM_API_KEY=your_key
export EMBEDDING_API_KEY=your_key
```

### Q6: 如何清理数据？

```bash
# 清理向量数据库
rm -rf data/db/

# 清理追踪数据
# 在 Dashboard 系统设置页面点击"清理追踪数据"
```

### Q7: 性能优化建议？

**摄取优化**：
- 增大 batch_size: `--batch-size 100`
- 禁用 Transform（如不需要）: `enable_transform=False`

**查询优化**：
- 减少 top_k: `--top-k 5`
- 禁用重排序（如不需要）: 不使用 `--rerank`

### Q8: 如何添加自定义 LLM？

```python
# src/libs/llm/your_provider.py
from .base import BaseLLM

class YourLLM(BaseLLM):
    def generate(self, prompt: str, **kwargs) -> str:
        # 实现生成逻辑
        pass
```

然后在 ComponentLoader 中注册。

---

## 🛠️ 开发指南

### 项目结构

```
RAG-MCP-SERVER/
├── src/
│   ├── config/              # 配置管理
│   ├── libs/                # 可插拔组件
│   ├── ingestion/           # 数据摄取
│   ├── retrieval/           # 检索系统
│   ├── mcp_server/          # MCP Server
│   ├── trace/               # 链路追踪
│   ├── dashboard/           # Dashboard
│   └── evaluation/          # 评估系统
├── scripts/                 # 脚本
├── tests/                   # 测试
├── config/                  # 配置文件
├── data/                    # 数据目录
└── README.md
```

### 添加新的 LLM 提供商

```python
# src/libs/llm/your_provider.py
from .base import BaseLLM

class YourLLM(BaseLLM):
    def __init__(self, config):
        self.api_key = config.get("your_provider.api_key")
        self.model = config.get("your_provider.model")
    
    def generate(self, prompt: str, **kwargs) -> str:
        # 实现生成逻辑
        response = your_api_call(prompt, **kwargs)
        return response.text
```

### 添加新的向量库

```python
# src/libs/vector_store/your_store.py
from .base import BaseVectorStore

class YourVectorStore(BaseVectorStore):
    def upsert(self, ids, texts, vectors, metadatas):
        # 实现上传逻辑
        pass
    
    def search(self, vector, top_k, filters=None):
        # 实现检索逻辑
        pass
```

### 代码风格

- 遵循 PEP 8
- 类名：PascalCase
- 函数名：snake_case
- 使用类型注解
- 添加 docstring

---

## 📄 许可证

MIT License

---

## 🤝 贡献

欢迎 PR 和 Issue！

### 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📧 联系

- 项目问题请提交 GitHub Issue
- 技术讨论欢迎在 Discussions 中交流

---

## 🙏 致谢

感谢以下开源项目：

- [Milvus](https://milvus.io/) - 向量数据库
- [Streamlit](https://streamlit.io/) - Dashboard 框架
- [Ragas](https://github.com/explodinggradients/ragas) - RAG 评估
- [LangChain](https://github.com/langchain-ai/langchain) - 灵感来源

---

**⭐ 如果这个项目对你有帮助，请给个 Star！**
