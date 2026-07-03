# 技术规格文档

> 版本：1.0 — 生产环境技术文档

## 目录

- [项目概述](#1-项目概述)
- [核心特点](#2-核心特点)
- [技术选型](#3-技术选型)
- [系统架构](#4-系统架构)
- [测试方案](#5-测试方案)
- [项目排期](#6-项目排期)
- [可扩展性展望](#7-可扩展性展望)

---

## 1. 项目概述

本项目基于多阶段检索增强生成（RAG）与模型上下文协议（MCP）设计，构建可扩展、高可观测、易迭代的智能问答与知识检索框架。

### 设计目标

- **可扩展性**：插拔式架构，支持组件无缝替换
- **可观测性**：全链路追踪，覆盖摄取和查询流程
- **生产就绪**：完整测试覆盖与质量保障机制
- **标准兼容**：完全符合 MCP 协议，融入生态系统

---

## 2. 核心特点

### 2.1 RAG 策略与设计亮点

项目在 RAG 链路的关键环节采用经典工程化优化策略：

- **分块策略**：智能分块与上下文增强
  - 语义感知切分，保留完整语义
  - 上下文增强：注入文档元数据（标题、页码）和图片描述
  
- **粗排召回（混合检索）**：第一阶段多模态召回
  - **稀疏检索（BM25）**：关键词精确匹配，解决专有名词查找
  - **稠密检索（Embedding）**：语义向量，解决同义词与模糊表达
  - **RRF 融合**：互惠排名融合算法，平衡查准率与查全率

- **精排重排（Rerank）**：深度语义排序
  - Cross-Encoder 或 LLM Rerank 识别细微语义差异
  - 两段式架构："粗排（低成本泛召回）→ 精排（高成本精过滤）"

### 2.2 全链路可插拔架构

系统每个核心环节定义抽象接口，支持"乐高积木式"组合：

- **LLM 调用层**：Azure OpenAI / OpenAI / Ollama / DeepSeek
- **Embedding & Rerank**：云端服务或本地模型
- **RAG 组件**：Loader / Splitter / Transformation 模块
- **检索策略**：向量 / 关键词 / 混合检索模式
- **评估体系**：Ragas / DeepEval / 自定义指标

配置驱动切换，零代码修改。

### 2.3 MCP 生态集成

完全遵循 Model Context Protocol (MCP) 标准：

- **零前端开发**：直接复用编辑器（VS Code）和 AI 助手
- **上下文互通**：Copilot 同时看到代码文件和知识库内容
- **标准兼容**：任何支持 MCP 的 AI Agent 可即刻接入

### 2.4 多模态图像处理

采用经典 **"图转文"（Image-to-Text）** 策略：

- **图像描述生成**：利用 Vision LLM 提取图像信息为文字描述
- **统一向量空间**：图像描述与文档文本一起向量化
- **优势**：架构统一、语义对齐、成本可控

### 2.5 可观测性与可视化管理

全链路透明可见且可管理：

- **全链路白盒化**：记录 Ingestion 和 Query 两条完整流水线的每个中间状态
- **可视化管理平台**：基于 Streamlit 的本地 Web 管理面板，提供六大功能页面
  - 系统总览 / 数据浏览器 / Ingestion 管理 / Ingestion 追踪 / Query 追踪 / 评估面板
- **自动化评估闭环**：集成 Ragas 等评估框架，建立数据驱动的迭代反馈回路

### 2.6 业务可扩展性

通用化架构设计，快速适配各类业务场景：

- **Agent 客户端扩展**：构建属于自己的 Agent 客户端
- **业务场景适配**：替换数据源，改造为私有知识库
- **检索逻辑定制**：调整检索策略以适配不同业务特点


---

## 3. 技术选型

### 3.1 RAG 核心流水线设计

#### 3.1.1 数据摄取流水线

**目标**：构建统一、可配置且可观测的数据摄取流水线，覆盖文档加载、格式解析、语义切分、多模态增强、嵌入计算、去重与批量上载。

**设计要点**：

- **明确分层职责**：
  - **Loader**：解析文件为统一 `Document` 对象（当前仅 PDF）
  - **Splitter**：基于 Markdown 结构切分 Document 为 Chunks
  - **Transform**：可插入处理步骤（ImageCaptioning、OCR、元数据增强）
  - **Embed & Upsert**：批量计算 embedding 并上载到向量存储

- **关键实现要素**：
  - **前置去重（文件完整性检查）**：SHA256 哈希指纹，未变更则零成本跳过
  - **解析与标准化**：PDF → Canonical Markdown（使用 MarkItDown）
  - **切分**：LangChain RecursiveCharacterTextSplitter，保持语义边界
  - **Transform & Enrichment（结构转换与深度增强）**：
    1. **智能重组**：LLM 对"粗切分"片段二次加工
    2. **语义元数据注入**：自动生成 Title、Summary、Tags
    3. **多模态增强**：Vision LLM 生成图片描述（Caption）
  - **Embedding（双路向量化）**：
    - **Dense Embeddings**：text-embedding-v4（2048维），语义理解
    - **Sparse Embeddings**：BM25 编码器，关键词精确匹配
  - **Upsert & Storage**：Milvus 双向量存储，原子化写入

**BM25 稀疏向量实现**：

**核心思想**：将传统 BM25 算法的输出转换为稀疏向量格式，直接存储到 Milvus。

**实现代码**：

```python
from collections import Counter
import math
import pickle

class BM25SparseEncoder:
    """BM25 稀疏向量编码器"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        参数说明：
        - k1: 词频饱和参数（1.2-2.0），控制词频的影响
        - b: 长度归一化参数（0-1），控制文档长度的影响
        """
        self.k1 = k1
        self.b = b
        self.vocab = {}        # term -> term_id 映射
        self.idf = {}          # term_id -> idf 值
        self.avgdl = 0         # 平均文档长度
        self.doc_count = 0     # 文档总数
    
    def fit(self, documents: list[str]):
        """在文档集上训练，构建词表和 IDF"""
        doc_lengths = []
        term_doc_freq = Counter()  # 每个词出现在多少个文档中
        
        # 统计词频和文档长度
        for doc in documents:
            terms = self._tokenize(doc)
            doc_lengths.append(len(terms))
            unique_terms = set(terms)
            
            # 为新词分配 ID
            for term in unique_terms:
                if term not in self.vocab:
                    self.vocab[term] = len(self.vocab)
                term_doc_freq[term] += 1
        
        self.doc_count = len(documents)
        self.avgdl = sum(doc_lengths) / self.doc_count if self.doc_count > 0 else 0
        
        # 计算 IDF：log((N - df + 0.5) / (df + 0.5) + 1)
        for term, term_id in self.vocab.items():
            df = term_doc_freq[term]
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
            self.idf[term_id] = idf
    
    def encode(self, text: str) -> dict:
        """将文本编码为稀疏向量"""
        terms = self._tokenize(text)
        term_freq = Counter(terms)
        doc_len = len(terms)
        
        indices = []
        values = []
        
        for term, tf in term_freq.items():
            if term in self.vocab:
                term_id = self.vocab[term]
                idf = self.idf[term_id]
                
                # BM25 公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score = idf * (numerator / denominator)
                
                indices.append(term_id)
                values.append(score)
        
        # 返回 Milvus 稀疏向量格式
        return {"indices": indices, "values": values}
    
    def _tokenize(self, text: str) -> list[str]:
        """分词（快速原型版：简单空格切分）"""
        # 生产环境应使用 jieba 等专业分词器
        return text.lower().split()
    
    def save(self, path: str):
        """持久化词表和 IDF"""
        with open(path, 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'idf': self.idf,
                'avgdl': self.avgdl,
                'doc_count': self.doc_count,
                'k1': self.k1,
                'b': self.b
            }, f)
    
    @classmethod
    def load(cls, path: str):
        """加载已训练的编码器"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        encoder = cls(k1=data['k1'], b=data['b'])
        encoder.vocab = data['vocab']
        encoder.idf = data['idf']
        encoder.avgdl = data['avgdl']
        encoder.doc_count = data['doc_count']
        return encoder
```

**使用流程**：

```python
# 1. 训练阶段（Ingestion Pipeline）
encoder = BM25SparseEncoder()
encoder.fit(all_chunk_texts)  # 在所有文档上训练
encoder.save('./data/db/bm25_encoder.pkl')  # 持久化

# 2. 编码阶段（对每个 chunk）
sparse_vector = encoder.encode(chunk_text)
# 输出示例: {"indices": [12, 45, 789], "values": [2.3, 1.8, 1.2]}

# 3. 查询阶段（Query Pipeline）
encoder = BM25SparseEncoder.load('./data/db/bm25_encoder.pkl')
query_sparse = encoder.encode(query_text)
```

**BM25 索引存储策略变更**：

**原设计 vs 新设计对比**：

| 维度 | 原设计（独立 BM25 索引） | 新设计（Milvus 统一存储） |
|------|----------------------|------------------------|
| **存储位置** | `data/db/bm25/` 独立文件系统 | Milvus `sparse_vector` 字段 |
| **索引结构** | 自定义倒排索引文件 | Milvus SPARSE_INVERTED_INDEX |
| **查询方式** | 自行实现 BM25 检索 | Milvus `search(sparse_vector)` |
| **同步问题** | 需手动同步两个存储系统 | ✅ 单一存储，无同步问题 |
| **维护成本** | 需维护两套存储逻辑 | ✅ 统一接口 |

**新设计优势**：

1. **统一存储**：Dense 和 Sparse 向量存储在同一 Collection，数据一致性有保障
2. **原生索引**：Milvus 的 `SPARSE_INVERTED_INDEX` 性能优于手工实现
3. **简化代码**：无需实现独立的 BM25 检索逻辑
4. **原子操作**：单次 `insert()` 同时写入两种向量，避免部分失败

**仍需保留的 BM25 组件**：

- ✅ `BM25SparseEncoder` 类：计算 IDF 和生成稀疏向量
- ✅ 词表持久化（`vocab` 和 `idf`）：查询时需要复用
- ❌ 独立的倒排索引文件：由 Milvus 替代
- ❌ BM25 检索逻辑：由 Milvus `search()` 替代

**数据目录结构调整**：

```
data/db/
├── milvus.db                    # Milvus Lite 数据文件（包含所有向量）
├── bm25_encoder.pkl             # BM25 编码器（vocab + idf）
├── ingestion_history.db         # 摄取历史（SQLite）
└── image_index.db               # 图片索引（SQLite）

# 不再需要：
# ├── bm25/                      # [删除] 独立 BM25 索引目录
# │   ├── inverted_index.json
# │   └── term_stats.json
```

**注意事项**：

1. **分词器选择**：
   - 快速原型：简单空格切分（`text.split()`）
   - 生产环境：使用 `jieba` 中文分词器
   
2. **词表管理**：
   - 增量摄取时，新词不会出现在 `vocab` 中（OOV 问题）
   - 解决方案：定期重新训练，或使用动态词表

3. **性能优化**：
   - 稀疏向量维度（词表大小）建议控制在 10万 以内
   - 可通过停用词过滤和低频词过滤减小词表

**Milvus 双向量存储设计**：

核心数据结构（Collection Schema）：

```python
# ChunkRecord 数据模型
{
    "id": "doc_abc_chunk_001",           # 主键，格式：{doc_id}_{chunk_index}
    "text": "原始文本内容...",            # 用于返回上下文
    "metadata": {                         # 动态字段，存储元数据
        "source": "document.pdf",
        "page": 5,
        "title": "章节标题",
        "doc_type": "technical"
    },
    "dense_vector": [0.1, 0.2, ...],     # Float Vector[2048]
    "sparse_vector": {                    # Sparse Float Vector
        "indices": [12, 45, 789, ...],   # 词项ID列表
        "values": [0.8, 0.6, 0.4, ...]   # BM25权重列表
    }
}
```

Milvus Collection 创建示例：

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient("./data/db/milvus.db")  # Lite模式：本地文件

# 定义 Schema
schema = client.create_schema(
    auto_id=False,
    enable_dynamic_field=True  # 支持metadata动态字段
)

# 添加字段
schema.add_field("id", DataType.VARCHAR, max_length=100, is_primary=True)
schema.add_field("text", DataType.VARCHAR, max_length=65535)
schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=2048)
schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

# 创建索引
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="dense_vector",
    index_type="AUTOINDEX",      # Lite模式自动选择最优索引
    metric_type="COSINE"         # 余弦相似度
)
index_params.add_index(
    field_name="sparse_vector",
    index_type="SPARSE_INVERTED_INDEX",  # 稀疏向量倒排索引
    metric_type="IP"             # 内积（稀疏向量标准）
)

# 创建 Collection
client.create_collection(
    collection_name="rag_knowledge_hub",
    schema=schema,
    index_params=index_params
)
```

**存储策略说明**：

| 特性 | 说明 | 优势 |
|------|------|------|
| **统一存储** | Dense + Sparse 向量存储在同一 Collection | 数据一致性，无需同步 |
| **动态元数据** | `enable_dynamic_field=True` | 灵活添加 metadata 字段 |
| **原子写入** | 单次 `insert()` 同时写入两种向量 | 保证数据完整性 |
| **索引优化** | Dense 用 AUTOINDEX，Sparse 用倒排索引 | 混合检索性能最优 |

#### 3.1.2 检索流水线

**目标**：实现核心 RAG 检索引擎，采用多阶段过滤架构。

- **Query Processing（查询预处理）**：关键词提取、查询扩展、Metadata 解析
- **Hybrid Search Execution（双路混合检索）**：
  - **Dense Route**：Query Embedding → 向量库（Cosine Similarity）
  - **Sparse Route**：BM25 → 倒排索引
  - **Fusion**：RRF 算法，基于排名倒数融合
- **Filtering & Reranking（精确过滤与重排）**：
  - **Metadata Filtering**：前置或后置过滤，基于索引能力
  - **Rerank Backend**：None / Cross-Encoder / LLM（可插拔，含回退）

**Milvus 混合检索实现方案**：

**方案对比**：

| 维度 | 方案A：分离查询 + 应用层融合 | 方案B：Milvus 内置混合检索 |
|------|---------------------------|--------------------------|
| **实现方式** | 分别调用 `search(dense)` 和 `search(sparse)`，用 Python 实现 RRF | 调用 `hybrid_search()` API，Milvus 内部融合 |
| **适用版本** | Milvus 2.3+ | Milvus 2.4+ |
| **控制灵活性** | ✅ 高，可自定义融合策略 | ⚠️ 中，依赖 Milvus 实现 |
| **性能** | ⚠️ 两次网络往返 | ✅ 一次网络往返 |
| **适用场景** | 快速原型、自定义融合算法 | 生产环境、标准 RRF |

**推荐：快速原型使用方案A，生产环境评估后选择方案B**

**方案A 实现示例**（推荐用于快速原型）：

```python
from pymilvus import MilvusClient

class HybridRetriever:
    def __init__(self, client: MilvusClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name
    
    def search(self, query_dense: list[float], query_sparse: dict, 
               top_k: int = 10, filters: str = None) -> list[dict]:
        """混合检索：Dense + Sparse + RRF 融合"""
        
        # 1. Dense 检索（扩大召回，取 top_k * 2）
        dense_results = self.client.search(
            collection_name=self.collection_name,
            data=[query_dense],
            anns_field="dense_vector",
            limit=top_k * 2,
            output_fields=["id", "text", "metadata"],
            filter=filters  # 例如: "metadata['source'] == 'doc1.pdf'"
        )
        
        # 2. Sparse 检索（扩大召回，取 top_k * 2）
        sparse_results = self.client.search(
            collection_name=self.collection_name,
            data=[query_sparse],
            anns_field="sparse_vector",
            limit=top_k * 2,
            output_fields=["id", "text", "metadata"],
            filter=filters
        )
        
        # 3. RRF 融合
        fused_results = self._rrf_fusion(
            dense_results[0],
            sparse_results[0],
            k=60  # RRF 参数，常用值 60
        )
        
        return fused_results[:top_k]
    
    def _rrf_fusion(self, dense_hits, sparse_hits, k: int = 60) -> list[dict]:
        """RRF (Reciprocal Rank Fusion) 算法"""
        scores = {}
        
        # Dense 路排名得分
        for rank, hit in enumerate(dense_hits, start=1):
            chunk_id = hit['id']
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
        
        # Sparse 路排名得分
        for rank, hit in enumerate(sparse_hits, start=1):
            chunk_id = hit['id']
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
        
        # 按融合得分排序
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # 构造结果（从原始结果中提取完整信息）
        id_to_hit = {}
        for hit in dense_hits + sparse_hits:
            if hit['id'] not in id_to_hit:
                id_to_hit[hit['id']] = hit
        
        return [
            {**id_to_hit[chunk_id], 'rrf_score': scores[chunk_id]}
            for chunk_id in sorted_ids
        ]
```

**元数据过滤示例**：

```python
# 按文档来源过滤
results = retriever.search(
    query_dense=dense_vec,
    query_sparse=sparse_vec,
    top_k=10,
    filters="metadata['source'] == 'technical_doc.pdf'"
)

# 复杂过滤条件（AND/OR）
filters = "(metadata['doc_type'] == 'manual') and (metadata['page'] > 10)"
```

**性能优化要点**：

| 优化项 | 说明 | 效果 |
|--------|------|------|
| **扩大召回** | 两路各取 `top_k * 2`，融合后取 `top_k` | 提升召回质量 |
| **并行查询** | 使用异步或多线程同时执行两路检索 | 减少延迟 50% |
| **元数据索引** | 对常用过滤字段创建标量索引 | 加速过滤 |
| **缓存查询向量** | 相同查询复用向量化结果 | 节省 API 调用 |

### 3.2 MCP 服务设计

**目标**：设计并实现符合 MCP 规范的 Server，作为知识上下文提供者。

#### 核心设计理念

- **协议优先**：严格遵循 MCP 官方规范（JSON-RPC 2.0）
- **开箱即用**：Client 端无需特殊配置
- **引用透明**：所有检索结果携带完整来源信息
- **多模态友好**：支持文本与图像等多种内容类型

#### 传输协议：Stdio

- **工作方式**：Client 以子进程启动 Server，通过标准输入/输出交换 JSON-RPC 消息
- **选型理由**：零配置、隐私安全、契合本地工作流
- **实现约束**：stdout 仅输出 MCP 消息，日志输出至 stderr

#### SDK：Python 官方 MCP SDK

- 官方维护，与协议规范同步更新
- 提供 `@server.tool()` 装饰器，声明式定义 Tools
- 内置 Stdio 与 HTTP Transport 支持

#### 对外暴露的工具函数

| 工具名称 | 功能描述 | 典型输入参数 | 输出特点 |
|---------|---------|-------------|---------|
| `query_knowledge_hub` | 主检索入口 | `query`, `top_k`, `collection` | 带引用的结构化结果 |
| `list_collections` | 列举知识库集合 | 无 | 集合名称、描述、文档数量 |
| `get_document_summary` | 获取文档摘要 | `doc_id` | 标题、摘要、创建时间、标签 |

### 3.3 可插拔架构设计

**目标**：定义清晰的抽象层与接口契约，使 RAG 链路每个核心组件可独立替换与升级。

#### 设计原则

- **接口隔离**：为每类组件定义最小化抽象接口
- **配置驱动**：通过统一配置文件指定各组件具体后端
- **工厂模式**：工厂函数根据配置动态实例化对应实现
- **优雅降级**：首选后端不可用时，自动回退到备选方案

#### 可插拔层级

| 抽象接口 | 当前默认实现 | 可替换选项 |
|---------|------------|----------|
| `LLMClient` | **Qwen Max (DashScope)** | Azure OpenAI / OpenAI / Ollama / DeepSeek / Qwen-Plus / Qwen-Turbo |
| `VisionLLMClient` | **Qwen-VL-Max (DashScope)** | Azure OpenAI Vision / OpenAI Vision / Ollama Vision / Qwen-VL-Plus |
| `EmbeddingClient` | **Qwen text-embedding-v4** | OpenAI / BGE / Ollama 本地模型 / text-embedding-v1/v2 |
| `Loader` | PDF Loader（MarkItDown） | Markdown/HTML/Code Loader |
| `FileIntegrity` | SQLite | Redis / PostgreSQL / JSON |
| `Splitter` | RecursiveCharacterTextSplitter | Semantic / FixedLength |
| `VectorStore` | **Milvus** | Chroma / Qdrant / Pinecone |
| `Reranker` | CrossEncoder | LLM Rerank / None |
| `Evaluator` | Ragas | DeepEval / 自定义指标 |

#### 千问（Qwen）模型推荐配置

**为什么选择千问模型？**

- **中文理解优势**：千问系列模型在中文语境下表现优异，特别适合中文文档的 RAG 应用
- **性价比高**：相比 GPT-4，千问模型提供相近的能力但成本更低
- **国内访问稳定**：通过阿里云 DashScope API 调用，国内访问延迟低、稳定性好
- **多模态支持**：Qwen-VL 系列在中文图表、截图识别方面表现出色

**推荐模型组合**：

| 使用场景 | 推荐模型 | 说明 |
|---------|---------|------|
| **文本生成（LLM）** | `qwen-max` ✅ | 最强能力，复杂推理和高质量生成（默认） |
| **文本生成（平衡）** | `qwen-plus` | 平衡性能与成本，适合大多数场景 |
| **文本生成（高速）** | `qwen-turbo` | 快速响应，适合实时交互场景 |
| **文本向量化（Embedding）** | `text-embedding-v4` ✅ | 最新版本，2048维向量，性能最优（默认） |
| **图片理解（Vision）** | `qwen-vl-max` ✅ | 中文图表识别准确，支持 OCR（默认） |
| **图片理解（快速）** | `qwen-vl-plus` | 速度更快，适合大批量图片处理 |

**API 调用说明**：

- **服务提供商**：阿里云 DashScope
- **文档地址**：https://help.aliyun.com/zh/dashscope/
- **API Key 获取**：https://dashscope.console.aliyun.com/apiKey
- **定价**：按 Token 计费，具体参考官方定价页面

#### Milvus 向量数据库配置（混合检索快速原型）

**为什么选择 Milvus？**

- **原生混合检索支持**：Milvus 2.4+ 原生支持稠密向量 + 稀疏向量混合检索
- **快速原型开发**：Milvus Lite 嵌入式部署，无需 Docker，类似 Chroma/SQLite
- **无缝扩展**：原型验证后可直接切换到 Milvus 标准版，无需修改代码
- **性能优越**：即使在快速原型模式下，性能也优于 Chroma

**快速原型部署（Milvus Lite）**：

```bash
# 安装 Milvus Lite（嵌入式版本）
pip install pymilvus

# 无需 Docker，直接在代码中使用
# 数据存储在本地文件：./data/db/milvus/milvus.db
```

**混合检索配置**：

```yaml
vector_store:
  backend: milvus
  uri: "./data/db/milvus/milvus.db"  # Lite 模式：本地文件路径
  # uri: "http://localhost:19530"    # 标准模式：Milvus 服务地址
  collection_name: rag_knowledge_hub
  
  # 混合检索配置（Dense + Sparse）
  dense_field: dense_vector         # 稠密向量字段名
  sparse_field: sparse_vector       # 稀疏向量字段名
  
  # 索引配置（自动创建）
  dense_index:
    type: AUTOINDEX                 # Lite 模式自动选择最优索引
    metric: COSINE                  # COSINE | L2 | IP
  sparse_index:
    type: SPARSE_INVERTED_INDEX     # 稀疏向量倒排索引
    metric: IP                      # 稀疏向量使用内积
```

**混合检索工作流程**：

1. **数据摄取**：同时存储 Dense Vector (2048维) + Sparse Vector (BM25)
2. **混合查询**：Milvus 内部并行执行稠密和稀疏检索
3. **融合排序**：使用 RRF（Reciprocal Rank Fusion）算法融合结果
4. **元数据过滤**：支持在检索时使用表达式过滤（collection, doc_type 等）

**快速原型 vs 生产部署**：

| 维度 | Milvus Lite（快速原型）| Milvus 标准版（生产） |
|------|---------------------|---------------------|
| **部署** | pip install，无需 Docker | Docker Compose / K8s |
| **数据规模** | < 100万向量 | 十亿级向量 |
| **性能** | 单机，适合开发测试 | 分布式，高并发 |
| **切换成本** | 仅修改 `uri` 配置 | 零代码修改 |

**相关文档**：
- Milvus Lite 快速开始：https://milvus.io/docs/milvus_lite.md
- 混合检索指南：https://milvus.io/docs/multi-vector-search.md
- Python SDK：https://github.com/milvus-io/pymilvus


### 3.4 可观测性与可视化管理平台设计

**目标**：全链路可观测追踪体系与完整可视化管理平台。

#### 设计理念

- **双链路全覆盖追踪**：Ingestion Trace 和 Query Trace 完整记录
- **透明可回溯**：每个阶段中间状态都被记录
- **低侵入性**：追踪逻辑与业务逻辑解耦
- **轻量本地化**：结构化日志 + 本地 Dashboard，零外部依赖
- **动态组件感知**：Dashboard 基于 Trace 字段动态渲染，自动适配组件变更

#### 追踪数据结构

**Query Trace（查询追踪）**：记录从 Query 输入到 Response 输出的全过程

**Ingestion Trace（摄取追踪）**：记录从文件加载到存储完成的全过程

#### 技术方案：结构化日志 + 本地 Web Dashboard

- **结构化日志层**：Python logging + JSON Formatter → JSON Lines 格式
- **本地 Web Dashboard**：基于 Streamlit 构建轻量级 Web UI
- **六大功能页面**：
  1. **系统总览**：组件配置卡片、数据资产统计
  2. **数据浏览器**：文档列表、Chunk 详情、图片预览
  3. **Ingestion 管理**：文件上传、摄取触发、文档删除
  4. **Ingestion 追踪**：摄取历史、阶段耗时瀑布图
  5. **Query 追踪**：查询历史、Dense/Sparse 对比、Rerank 变化
  6. **评估面板**：运行评估、查看指标、历史趋势

### 3.5 多模态图片处理设计

**目标**：完整的图片处理方案，实现"用自然语言搜索图片"的能力。

#### 策略选型：Image-to-Text（图转文）

- **图像描述生成**：Vision LLM 将图片转化为文本描述
- **统一向量空间**：描述文本与文档文本一起向量化
- **优势**：架构统一、语义对齐、成本可控

#### 图片处理全流程

```
原始文档 (PDF/PPT/Markdown)
    ↓
Loader 阶段：图片提取与引用收集
    ↓
Splitter 阶段：保持图文关联
    ↓
Transform 阶段：图片理解与描述生成
    ↓
Storage 阶段：双轨存储（向量库 + 文件系统）
```

#### Vision LLM 选型

| 模型 | 提供商 | 特点 | 适用场景 | 推荐指数 |
|-----|--------|------|---------|---------|
| **GPT-4o** | OpenAI / Azure | 理解能力强，复杂图表解读准确 | 高质量需求、英文文档 | ⭐⭐⭐⭐⭐ |
| **Qwen-VL-Max** | 阿里云 | 中文理解出色，性价比高 | 中文文档、国内部署 | ⭐⭐⭐⭐⭐ |
| **Qwen-VL-Plus** | 阿里云 | 速度更快，成本更低 | 大批量中文文档 | ⭐⭐⭐⭐ |
| **Claude 3.5 Sonnet** | Anthropic | 多模态原生，长上下文 | 结合大段文字理解图片 | ⭐⭐⭐⭐ |


---

## 4. 系统架构

### 4.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Clients（外部调用层）                  │
│   GitHub Copilot / Claude Desktop / 其他 MCP Agent          │
│                    JSON-RPC 2.0 (Stdio)                      │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    MCP Server 层（接口层）                    │
│   Protocol Handler → Tools (query/list/get_summary)         │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     Core 层（核心业务逻辑）                   │
│   Query Engine: Processor → Hybrid Search → Reranker        │
│   Response Builder: Citation → Multimodal Assembly          │
│   Trace Collector: Context → JSON Lines                     │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     Storage 层（存储层）                      │
│   Vector Store (Chroma) / BM25 Index / Image Store          │
│   Trace Logs (JSON Lines) / Processing Cache                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              Ingestion Pipeline（离线数据摄取）               │
│   Loader → Splitter → Transform → Embedding → Upsert        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              Libs 层（可插拔抽象层）                          │
│   LLM / Embedding / Splitter / VectorStore /                │
│   Reranker / Evaluator Factories                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              Observability 层（可观测性）                     │
│   TraceContext / Dashboard (Streamlit) /                    │
│   Evaluation Module / Structured Logger                     │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 目录结构

```
smart-knowledge-hub/
│
├── config/                    # 配置文件目录
│   ├── settings.yaml          # 主配置文件
│   └── prompts/               # Prompt 模板目录
│
├── src/                       # 源代码主目录
│   ├── mcp_server/            # MCP Server 层
│   ├── core/                  # Core 层（核心业务逻辑）
│   │   ├── query_engine/      # 查询引擎模块
│   │   ├── response/          # 响应构建模块
│   │   └── trace/             # 追踪模块
│   ├── ingestion/             # Ingestion Pipeline
│   ├── libs/                  # Libs 层（可插拔抽象层）
│   │   ├── loader/
│   │   ├── llm/
│   │   ├── embedding/
│   │   ├── splitter/
│   │   ├── vector_store/
│   │   ├── reranker/
│   │   └── evaluator/
│   └── observability/         # Observability 层
│       ├── dashboard/         # Web Dashboard
│       └── evaluation/        # 评估模块
│
├── data/                      # 数据目录
│   ├── documents/             # 原始文档存放
│   ├── images/                # 提取的图片存放
│   └── db/                    # 数据库与索引文件
│       ├── ingestion_history.db
│       ├── image_index.db
│       ├── chroma/
│       └── bm25/
│
├── logs/                      # 日志目录
│   └── traces.jsonl           # 追踪日志（JSON Lines）
│
├── tests/                     # 测试目录
│   ├── unit/                  # 单元测试
│   ├── integration/           # 集成测试
│   ├── e2e/                   # 端到端测试
│   └── fixtures/              # 测试数据
│
├── scripts/                   # 脚本目录
│   ├── ingest.py              # 数据摄取脚本
│   ├── query.py               # 查询测试脚本
│   ├── evaluate.py            # 评估运行脚本
│   └── start_dashboard.py     # Dashboard 启动脚本
│
├── main.py                    # MCP Server 启动入口
├── pyproject.toml
├── requirements.txt
└── README.md
```


### 4.3 配置管理示例

```yaml
# config/settings.yaml

# LLM 配置
llm:
  provider: dashscope          # azure | openai | ollama | deepseek | dashscope
  model: qwen-max              # qwen-max（默认）| qwen-plus | qwen-turbo
  api_key: "${DASHSCOPE_API_KEY}"  # 从环境变量获取

# Embedding 配置
embedding:
  provider: dashscope
  model: text-embedding-v4     # text-embedding-v4（默认，2048维）| v2 | v1
  api_key: "${DASHSCOPE_API_KEY}"
  
# Vision LLM 配置（图片描述生成）
vision_llm:
  provider: dashscope
  model: qwen-vl-max           # qwen-vl-max（默认）| qwen-vl-plus
  api_key: "${DASHSCOPE_API_KEY}"
  
# 向量存储配置
vector_store:
  backend: milvus              # milvus（默认）| chroma | qdrant | pinecone
  host: localhost
  port: 19530
  collection_name: rag_knowledge_hub

# 检索配置
retrieval:
  sparse_backend: bm25
  fusion_algorithm: rrf
  top_k_dense: 20
  top_k_sparse: 20
  top_k_final: 10

# 重排配置
rerank:
  backend: cross_encoder       # none | cross_encoder | llm
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
  top_m: 30

# 评估配置
evaluation:
  backends: [ragas, custom]
  golden_test_set: ./tests/fixtures/golden_test_set.json

# 可观测性配置
observability:
  enabled: true
  log_file: ./logs/traces.jsonl

# Dashboard 配置
dashboard:
  enabled: true
  port: 8501
  traces_dir: ./logs
  auto_refresh: true
  refresh_interval: 5
```

### 4.4 环境变量配置

**API Key 安全最佳实践**：所有敏感信息（API Key、密钥等）均通过环境变量注入，避免硬编码到配置文件中。

#### 配置方式

**Linux / macOS**:
```bash
# 在 ~/.bashrc 或 ~/.zshrc 中添加
export DASHSCOPE_API_KEY="your-api-key-here"

# 使配置生效
source ~/.bashrc
```

**Windows (PowerShell)**:
```powershell
# 临时设置（当前会话有效）
$env:DASHSCOPE_API_KEY = "your-api-key-here"

# 永久设置（系统环境变量）
[System.Environment]::SetEnvironmentVariable('DASHSCOPE_API_KEY', 'your-api-key-here', 'User')
```

**使用 .env 文件（开发环境推荐）**:
```bash
# 在项目根目录创建 .env 文件（确保已加入 .gitignore）
DASHSCOPE_API_KEY=your-api-key-here
```

然后在代码中使用 `python-dotenv` 库加载：
```python
from dotenv import load_dotenv
load_dotenv()  # 自动加载 .env 文件到环境变量
```

#### 多模型 API Key 管理

如果同时使用多个服务提供商，可以配置多个环境变量：

```bash
# 千问/通义千问（阿里云 DashScope）
export DASHSCOPE_API_KEY="sk-xxx"

# OpenAI
export OPENAI_API_KEY="sk-xxx"

# Azure OpenAI
export AZURE_OPENAI_API_KEY="xxx"
export AZURE_OPENAI_ENDPOINT="https://xxx.openai.azure.com/"

# Ollama（本地部署，无需 API Key）
export OLLAMA_BASE_URL="http://localhost:11434"
```

#### 配置文件中引用环境变量

在 `settings.yaml` 中使用 `${VAR_NAME}` 语法引用环境变量：

```yaml
llm:
  provider: dashscope
  api_key: "${DASHSCOPE_API_KEY}"  # 运行时自动替换为环境变量的值
```

系统会在加载配置时自动展开环境变量，如果环境变量不存在，会抛出明确的错误提示。

---

## 5. 测试方案

### 5.1 测试金字塔

```
        /\
       /E2E\         ← 少量，验证关键业务流程
      /------\
     /Integration\   ← 中量，验证模块协作
    /------------\
   /  Unit Tests  \  ← 大量，验证单个函数/类
  /________________\
```

### 5.2 测试分层策略

#### 单元测试 (Unit Tests)

**目标**：验证每个独立组件的内部逻辑正确性。

**覆盖范围**：
- Loader / Splitter / Transform / Embedding / Retrieval / Reranker
- Mock 外部依赖（LLM API、Vector Store）
- 快速反馈（<1秒每个测试）

**技术选型**：
- 测试框架：`pytest`
- Mock 工具：`unittest.mock` / `pytest-mock`
- 断言增强：`pytest-check`

#### 集成测试 (Integration Tests)

**目标**：验证多个组件协作时的数据流转与接口兼容性。

**覆盖范围**：
- Ingestion Pipeline（Loader → Splitter → Transform → Storage）
- Hybrid Search（Dense + Sparse → Fusion）
- Rerank Pipeline（召回 → 过滤 → 重排）
- MCP Server（端到端工具调用）

**技术选型**：
- 数据隔离：独立临时数据库/向量库
- 异步测试：`pytest-asyncio`
- 契约测试：定义模块间 Schema

#### 端到端测试 (End-to-End Tests)

**目标**：模拟真实用户操作，验证完整业务流程可用性。

**核心场景**：
1. **数据准备（离线摄取）**：验证文档摄取流程完整性与正确性
2. **召回测试**：验证检索系统召回精度与排序质量
3. **MCP Client 功能测试**：验证 MCP Server 与 Client 协议兼容性

**测试工具**：
- BDD 框架：`behave` 或 `pytest-bdd`
- 环境准备：临时测试向量库、预置标准测试文档集

### 5.3 质量保障

**测试覆盖率目标**：
- 单元测试：核心逻辑覆盖率 ≥ 80%
- 集成测试：关键路径覆盖率 100%
- E2E 测试：核心用户场景覆盖率 100%

**关键指标达标验证**：
- 检索指标：Hit Rate@K ≥ 90%、MRR ≥ 0.8、NDCG@K ≥ 0.85
- 生成指标：Faithfulness ≥ 0.9、Answer Relevancy ≥ 0.85


---

## 6. 项目排期

### 阶段总览

1. **阶段 A：工程骨架与测试基座** - 建立可运行、可配置、可测试的工程骨架
2. **阶段 B：Libs 可插拔层** - Factory + Base 接口 + 默认可运行实现
3. **阶段 C：Ingestion Pipeline MVP** - PDF→MD→Chunk→Embedding→Upsert
4. **阶段 D：Retrieval MVP** - Dense + Sparse + RRF + 可选 Rerank
5. **阶段 E：MCP Server 层与 Tools** - 按 MCP 标准暴露 tools
6. **阶段 F：Trace 基础设施与打点** - Ingestion + Query 双链路可追踪
7. **阶段 G：可视化管理平台 Dashboard** - 搭建 Streamlit 六页面管理平台
8. **阶段 H：评估体系** - RagasEvaluator + CompositeEvaluator + Golden test set
9. **阶段 I：端到端验收与文档收口** - 完整测试与文档完善

### 进度跟踪

| 阶段 | 总任务数 | 已完成 | 进度 |
|------|---------|--------|------|
| 阶段 A | 3 | 0 | 0% |
| 阶段 B | 16 | 0 | 0% |
| 阶段 C | 15 | 0 | 0% |
| 阶段 D | 7 | 0 | 0% |
| 阶段 E | 6 | 0 | 0% |
| 阶段 F | 5 | 0 | 0% |
| 阶段 G | 6 | 0 | 0% |
| 阶段 H | 5 | 0 | 0% |
| 阶段 I | 5 | 0 | 0% |
| **总计** | **68** | **0** | **0%** |

### 核心阶段说明

#### 阶段 A：工程骨架与测试基座

**目标**：先可导入，再可测试

- 初始化目录树与最小可运行入口
- 引入 pytest 并建立测试目录约定
- 配置加载与校验（Settings）

#### 阶段 B：Libs 可插拔层

**目标**：Factory 可工作，且至少有"默认后端"可跑通端到端

- LLM / Embedding / Splitter / VectorStore / Reranker / Evaluator 抽象接口与工厂
- 补齐默认实现（OpenAI/Azure/Ollama LLM，OpenAI/Azure/Ollama Embedding，RecursiveSplitter，ChromaStore，CrossEncoder/LLM Reranker）
- Vision LLM 抽象接口与 Azure Vision 实现

#### 阶段 C：Ingestion Pipeline MVP

**目标**：能把 PDF 样例摄取到本地存储

- 定义核心数据类型/契约（Document/Chunk/ChunkRecord）
- 文件完整性检查（SHA256）
- Loader 抽象基类与 PDF Loader
- Splitter 集成（调用 Libs）
- Transform（ChunkRefiner / MetadataEnricher / ImageCaptioner）
- Embedding（DenseEncoder / SparseEncoder / BatchProcessor）
- Storage（BM25Indexer / VectorUpserter / ImageStorage）
- Pipeline 编排与脚本入口

#### 阶段 D：Retrieval MVP

**目标**：能 query 并返回 Top-K chunks

- QueryProcessor（关键词提取 + filters）
- DenseRetriever / SparseRetriever
- RRF Fusion
- HybridSearch 编排
- Reranker（Core 层编排 + Fallback）
- 脚本入口 query.py

#### 阶段 E：MCP Server 层与 Tools

**目标**：对外可用的 MCP tools

- MCP Server 入口与 Stdio 约束
- Protocol Handler 协议解析与能力协商
- 实现 tools：query_knowledge_hub / list_collections / get_document_summary
- 多模态返回组装（Text + Image）

#### 阶段 F：Trace 基础设施与打点

**目标**：Ingestion + Query 双链路可追踪

- TraceContext 增强（finish + 耗时统计 + trace_type）
- 结构化日志 logger（JSON Lines）
- 在 Query 链路打点
- 在 Ingestion 链路打点
- Pipeline 进度回调 (on_progress)

#### 阶段 G：可视化管理平台 Dashboard

**目标**：六页面完整可视化管理

- Dashboard 基础架构与系统总览页
- DocumentManager 实现
- 数据浏览器页面
- Ingestion 管理页面
- Ingestion 追踪页面
- Query 追踪页面

#### 阶段 H：评估体系

**目标**：可插拔评估 + 可量化回归

- RagasEvaluator 实现
- CompositeEvaluator 实现
- EvalRunner + Golden Test Set
- 评估面板页面
- Recall 回归测试（E2E）

#### 阶段 I：端到端验收与文档收口

**目标**：开箱即用的"可复现"工程

- E2E：MCP Client 侧调用模拟
- E2E：Dashboard 冒烟测试
- 完善 README（运行说明 + 测试说明 + MCP 配置 + Dashboard 使用）
- 清理接口一致性（契约测试补齐）
- 全链路 E2E 验收

### 交付里程碑

- **M1（完成阶段 A+B）**：工程骨架 + 可插拔抽象层就绪
- **M2（完成阶段 C）**：离线摄取链路可用
- **M3（完成阶段 D+E）**：在线查询 + MCP tools 可用
- **M4（完成阶段 F）**：双链路追踪，JSON Lines 持久化
- **M5（完成阶段 G）**：六页面可视化管理平台就绪
- **M6（完成阶段 H+I）**：评估体系完整 + E2E 验收通过 + 文档完善


---

## 7. 可扩展性展望

### 7.1 云端部署与后端架构

架构设计完全支持向云端迁移：

- **Server 容器化**：编写 Dockerfile，将 MCP Server 打包为容器，深入理解 Python 环境隔离、依赖管理与 Docker 最佳实践
- **云端接入**：部署至 Azure Container Apps 或 AWS Lambda
  - 处理网络延时、配置 API Gateway、增加 AuthN/AuthZ 鉴权机制
- **多租户与并发**：从单用户本地服务转变为支持团队共享
  - 在 Chroma 中实现 Namespace 隔离、处理并发请求锁、优化 embedding 缓存策略

### 7.2 业务深耕：从"通用"到"垂直"

将通用技术框架与具体业务场景深度结合：

#### 多源异构数据的复杂适配

现实业务中不仅有 PDF，还大量存在 PPTX, DOCX, XLSX 甚至 HTML 数据。

**挑战**：
- 如何处理不同格式的特有语义？
- PPT 中的演讲者备注往往比正文更关键
- Excel 中的公式逻辑与跨行关联如何保留？

#### 复杂结构化数据的精确理解

简单的文本切分在处理表格、层级列表时往往会破坏语义。

**挑战**：
- **表格理解**：如何处理跨页长表格、合并单元格以及含有复杂表头的财务报表？
- **上下文断裂**：当一个完整逻辑段落被切分到两个 chunk 时，如何保证检索时能感知整体上下文？

#### 业务逻辑驱动的生成控制

仅仅根据"相似度"召回文档在企业级场景中往往不够。

**挑战**：
- **时效性与版本管理**：当知识库中同时存在"2023版"和"2024版"规章时，如何确保系统不会混淆历史数据与最新标准？
- **权限与受众适配**：面对内部员工与外部客户，如何控制生成答案的详略程度与敏感信息披露？
- **拒答机制**：当召回内容的置信度不足时，如何让系统诚实地回答"不知道"而不是强行拼凑答案？

### 7.3 迈向自主智能：Agentic RAG 的演进路径

当前的 RAG 架构主要遵循"一次检索-一次生成"的固有范式，但在面对极其复杂的问题时，单一的线性流程往往力不从心。本项目作为标准的 MCP Server，天然具备向 **Agentic RAG（代理式 RAG）** 演进的潜力。

#### 从"单步检索"到"多步决策"

**当前**：Agent 只调用一个通用的 `search` 工具

**演进方向**：Server 暴露更原子化的工具
- `list_directory`（查看目录结构）
- `preview_document`（预览摘要）
- `verify_fact`（事实核查）

Agent 可以像人类研究员一样，先看目录圈定范围，再针对性阅读，最后交叉验证信息。

#### 让 Agent 具备"反思"能力

**演进方向**：利用现有的评估模块，Server 提供 `self_check` 接口

Agent 在生成答案后，可以自主调用该接口检测是否存在幻觉，或者检索结果是否真正支撑了论点。如果发现不足，Agent 可以自主决定进行第二轮更深度的搜索。

#### 动态策略选择

**演进方向**：不再硬编码使用混合检索

Server 将 `keyword_search` 和 `semantic_search` 作为独立工具暴露。Agent 可以根据用户意图自主判断：
- 如果是搜人名，只用关键词搜
- 如果是搜概念，通过语义搜

这种工具使用的灵活性正是 Agentic RAG 的核心魅力。

---

**文档版本**：1.1  
**最后更新**：2026-07-02  
**项目状态**：待开发（0/68 任务完成）

**技术栈配置**：
- 大语言模型：Qwen-Max (DashScope)
- 向量模型：text-embedding-v4 (2048维)
- 视觉模型：Qwen-VL-Max (DashScope)
- 向量数据库：Milvus Lite（快速原型，支持混合检索）
