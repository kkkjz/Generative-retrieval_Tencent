# MINDER 多视角生成式检索模型训练指南

本指南提供了将自定义JSON数据转换为MINDER项目格式并训练基础多视角生成式检索模型的完整流程。

## 项目概述

MINDER是一个多视角标识符增强的生成式检索系统，支持从多个视角（Title、Span、Query）进行训练，实现高效的文档检索。

## 环境要求

- Python 3.7+
- CUDA支持的GPU
- 至少16GB内存
- 足够的磁盘空间（建议50GB+）

## 快速开始

### 1. 准备MINDER项目

首先克隆MINDER项目：

```bash
cd /root/autodl-tmp
git clone https://github.com/microsoft/MINDER.git
```

### 2. 准备数据

您的JSON数据应该遵循以下格式：

```json
{
  "查询文本": {
    "docid": "文档ID",
    "title": "文档标题",
    "doc_content": "文档内容（用于生成span，不超过2048字符）",
    "pseudo_query": ["伪查询1", "伪查询2", ...]
  }
}
```

我们提供了一个示例数据文件 `sample_data.json`，您可以参考其格式。

### 3. 运行数据转换和训练

#### 步骤1：运行数据转换脚本

```bash
python3 data_converter_and_trainer.py sample_data.json --output_dir ./minder_training
```

这将创建以下目录结构：
```
minder_training/
├── data/
│   ├── custom_corpus.tsv          # 语料库文件
│   ├── custom-train.json          # 训练集
│   ├── custom-dev.json            # 验证集
│   ├── training_data/
│   │   └── custom_dataset/         # 处理后的训练数据
│   ├── fm_index/
│   │   └── custom/                 # FM-index文件
│   └── pseudo_queries/
│       └── pid2query_custom.pkl    # 伪查询映射
├── setup_environment.sh           # 环境设置脚本
├── preprocess_custom.sh           # 数据预处理脚本
├── train_custom_model.sh          # 模型训练脚本
└── inference_custom_model.sh      # 推理脚本
```

#### 步骤2：设置环境

```bash
cd minder_training
bash setup_environment.sh
```

#### 步骤3：下载BART模型

手动下载BART-large模型到指定目录：

```bash
# 创建目录
mkdir -p MINDER/res/external/bart_large

# 下载模型文件（需要根据fairseq文档下载）
# 下载地址：https://github.com/pytorch/fairseq/blob/main/examples/bart/README.md
```

#### 步骤4：预处理数据

```bash
bash preprocess_custom.sh
```

#### 步骤5：开始训练

```bash
bash train_custom_model.sh
```

训练过程可能需要几个小时到几天，取决于数据大小和硬件配置。

#### 步骤6：运行推理（可选）

首先创建测试查询文件，然后运行推理：

```bash
# 创建测试查询文件（CSV格式）
echo "query_id,query" > data/test_queries.csv
echo "1,什么是人工智能？" >> data/test_queries.csv

# 运行推理
bash inference_custom_model.sh
```

## 数据格式说明

### 输入JSON格式

- **query**: 查询文本，长度不超过2048字符
- **docid**: 文档的唯一标识符
- **title**: 文档标题
- **doc_content**: 文档内容，用于生成span，长度不超过2048字符
- **pseudo_query**: 伪查询列表，用于无监督训练

### 多视角训练

脚本会自动生成三种视角的训练数据：

1. **Title视角**: 从文档标题生成训练样本
2. **Span视角**: 从文档内容中提取相关片段
3. **Query视角**: 使用伪查询进行训练

## 训练参数说明

主要训练参数（在 `train_custom_model.sh` 中）：

- `--max-tokens 4096`: 批次大小
- `--max-update 100000`: 最大更新步数
- `--lr 3e-05`: 学习率
- `--warmup-updates 500`: 预热步数
- `--save-interval-updates 5000`: 保存间隔

您可以根据需要调整这些参数。

## 故障排除

### 常见问题

1. **内存不足**: 减少 `--max-tokens` 参数
2. **CUDA错误**: 检查GPU驱动和CUDA版本
3. **依赖缺失**: 确保所有Python包都已安装
4. **BART模型缺失**: 确保已正确下载BART模型

### 日志查看

训练日志会保存在checkpoints目录中，可以使用以下命令查看：

```bash
tail -f checkpoints/train.log
```

## 性能优化建议

1. **数据大小**: 建议至少1000条训练样本
2. **硬件配置**: 使用V100或更高级的GPU
3. **批次大小**: 根据GPU内存调整max-tokens
4. **训练时间**: 根据数据大小调整max-update

## 模型评估

训练完成后，可以使用验证集评估模型性能：

```bash
# 在验证集上运行推理
python MINDER/seal/search.py \
    --topics data/custom-dev.json \
    --topics_format dpr_qas \
    --output results/dev_results.json \
    --checkpoint checkpoints/checkpoint_best.pt \
    --fm_index data/fm_index/custom/custom_corpus.fm_index
```

## 进阶使用

### 自定义训练参数

您可以修改 `train_custom_model.sh` 中的参数来优化训练：

- 调整学习率和训练步数
- 修改批次大小和梯度累积
- 添加正则化参数

### 增量训练

如果需要在现有模型基础上继续训练：

```bash
# 修改训练脚本，使用已有checkpoint
--restore-file checkpoints/checkpoint_last.pt
```

## 技术支持

如果遇到问题，请检查：

1. MINDER项目的官方文档
2. fairseq的安装和使用指南
3. 确保所有依赖都已正确安装

## 许可证

本脚本基于MINDER项目，请遵循相应的开源许可证。