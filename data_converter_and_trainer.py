#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据转换和训练脚本
将用户的JSON格式数据转换为MINDER项目所需的格式，并实现一键训练基础多视角生成式检索模型
"""

import json
import os
import csv
import pickle
import argparse
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import multiprocessing
from tqdm import tqdm
from fuzzywuzzy import fuzz
import math

class MINDERDataConverter:
    def __init__(self, input_json_path: str, output_dir: str):
        self.input_json_path = input_json_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建必要的子目录
        self.data_dir = self.output_dir / "data"
        self.training_data_dir = self.data_dir / "training_data" / "custom_dataset"
        self.fm_index_dir = self.data_dir / "fm_index" / "custom"
        self.pseudo_queries_dir = self.data_dir / "pseudo_queries"
        
        for dir_path in [self.data_dir, self.training_data_dir, self.fm_index_dir, self.pseudo_queries_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_data(self) -> Dict[str, Any]:
        """加载用户的JSON数据"""
        print("正在加载数据...")
        with open(self.input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"加载了 {len(data)} 条数据")
        return data
    
    def create_corpus_file(self, data: Dict[str, Any]):
        """创建语料库文件 (TSV格式)"""
        print("正在创建语料库文件...")
        corpus_file = self.data_dir / "custom_corpus.tsv"
        
        with open(corpus_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='\t', quotechar='"')
            # 写入标题行
            writer.writerow(["id", "text", "title"])
            
            for query, doc_info in data.items():
                docid = doc_info["docid"]
                title = doc_info["title"]
                content = doc_info["doc_content"]
                
                # 限制内容长度到2048字符
                if len(content) > 2048:
                    content = content[:2048]
                
                writer.writerow([docid, content, title])
        
        print(f"语料库文件已创建: {corpus_file}")
        return corpus_file
    
    def create_dpr_format_data(self, data: Dict[str, Any]):
        """创建DPR格式的训练和验证数据"""
        print("正在创建DPR格式数据...")
        
        # 分割训练集和验证集 (80:20)
        queries = list(data.keys())
        random.shuffle(queries)
        split_idx = int(len(queries) * 0.8)
        train_queries = queries[:split_idx]
        dev_queries = queries[split_idx:]
        
        # 创建训练集
        train_data = []
        for query in train_queries:
            doc_info = data[query]
            train_data.append({
                "question": query[:2048],  # 限制查询长度
                "answers": [query],  # 使用查询本身作为答案
                "positive_ctxs": [{
                    "title": doc_info["title"],
                    "text": doc_info["doc_content"][:2048],  # 限制内容长度
                    "score": 1000,
                    "title_score": 1,
                    "passage_id": doc_info["docid"],
                    "psg_id": doc_info["docid"]
                }],
                "negative_ctxs": [],
                "hard_negative_ctxs": []
            })
        
        # 创建验证集
        dev_data = []
        for query in dev_queries:
            doc_info = data[query]
            dev_data.append({
                "question": query[:2048],
                "answers": [query],
                "positive_ctxs": [{
                    "title": doc_info["title"],
                    "text": doc_info["doc_content"][:2048],
                    "score": 1000,
                    "title_score": 1,
                    "passage_id": doc_info["docid"],
                    "psg_id": doc_info["docid"]
                }],
                "negative_ctxs": [],
                "hard_negative_ctxs": []
            })
        
        # 保存文件
        train_file = self.data_dir / "custom-train.json"
        dev_file = self.data_dir / "custom-dev.json"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(dev_file, 'w', encoding='utf-8') as f:
            json.dump(dev_data, f, ensure_ascii=False, indent=2)
        
        print(f"训练集已创建: {train_file} ({len(train_data)} 条)")
        print(f"验证集已创建: {dev_file} ({len(dev_data)} 条)")
        
        return train_file, dev_file
    
    def create_pseudo_queries_mapping(self, data: Dict[str, Any]):
        """创建伪查询映射文件"""
        print("正在创建伪查询映射...")
        
        pid2query = {}
        for query, doc_info in data.items():
            docid = doc_info["docid"]
            pseudo_queries = doc_info["pseudo_query"]
            # 确保至少有一个伪查询
            if not pseudo_queries:
                pseudo_queries = [query[:2048]]
            else:
                # 限制伪查询长度
                pseudo_queries = [pq[:2048] for pq in pseudo_queries]
            
            pid2query[str(docid)] = pseudo_queries
        
        # 保存pickle文件
        pid2query_file = self.pseudo_queries_dir / "pid2query_custom.pkl"
        with open(pid2query_file, 'wb') as f:
            pickle.dump(pid2query, f)
        
        print(f"伪查询映射已创建: {pid2query_file}")
        return pid2query_file
    
    def generate_supervised_data(self, train_file: str, dev_file: str, pid2query_file: str):
        """生成监督训练数据"""
        print("正在生成监督训练数据...")
        
        # 生成三种视角的训练数据
        targets = ["title", "span", "query"]
        n_samples = {"title": 3, "span": 10, "query": 5}
        
        for split in ["train", "dev"]:
            input_file = train_file if split == "train" else dev_file
            
            for target in targets:
                print(f"生成 {split} 集的 {target} 数据...")
                
                cmd = [
                    "python3", "MINDER/scripts/training/make_supervised_dpr_dataset.py",
                    str(input_file),
                    str(self.training_data_dir / split),
                    "--target", target,
                    "--mark_target",
                    "--mark_silver",
                    "--n_samples", str(n_samples[target]),
                    "--mode", "a" if target != "title" else "w",
                    "--min_score", "0.0",
                    "--min_score_gold", "0.0"
                ]
                
                if target == "query":
                    cmd.extend(["--pid2query", str(pid2query_file)])
                
                try:
                    subprocess.run(cmd, check=True, cwd=self.output_dir)
                except subprocess.CalledProcessError as e:
                    print(f"生成 {target} 数据时出错: {e}")
                    raise
    
    def generate_unsupervised_data(self, corpus_file: str, pid2query_file: str):
        """生成无监督训练数据"""
        print("正在生成无监督训练数据...")
        
        cmd = [
            "python3", "MINDER/scripts/training/make_generated_dataset2.py",
            str(corpus_file),
            str(self.training_data_dir / "unsupervised.source"),
            str(self.training_data_dir / "unsupervised.target"),
            "--format", "dpr",
            "--num_samples", "3",
            "--num_title_samples", "1",
            "--num_query_samples", "2",
            "--full_doc_n", "1",
            "--mark_pretraining",
            "--pid2query", str(pid2query_file)
        ]
        
        try:
            subprocess.run(cmd, check=True, cwd=self.output_dir)
        except subprocess.CalledProcessError as e:
            print(f"生成无监督数据时出错: {e}")
            raise
    
    def merge_training_data(self):
        """合并监督和无监督训练数据"""
        print("正在合并训练数据...")
        
        # 合并source文件
        with open(self.training_data_dir / "train.source", 'a', encoding='utf-8') as target_file:
            with open(self.training_data_dir / "unsupervised.source", 'r', encoding='utf-8') as source_file:
                target_file.write(source_file.read())
        
        # 合并target文件
        with open(self.training_data_dir / "train.target", 'a', encoding='utf-8') as target_file:
            with open(self.training_data_dir / "unsupervised.target", 'r', encoding='utf-8') as source_file:
                target_file.write(source_file.read())
        
        print("训练数据合并完成")
    
    def build_fm_index(self, corpus_file: str):
        """构建FM-index"""
        print("正在构建FM-index...")
        
        cmd = [
            "python3", "MINDER/scripts/build_fm_index.py",
            str(corpus_file),
            str(self.fm_index_dir / "custom_corpus.fm_index")
        ]
        
        try:
            subprocess.run(cmd, check=True, cwd=self.output_dir)
        except subprocess.CalledProcessError as e:
            print(f"构建FM-index时出错: {e}")
            raise
        
        print(f"FM-index已构建: {self.fm_index_dir / 'custom_corpus.fm_index'}")
    
    def preprocess_for_fairseq(self):
        """为fairseq预处理数据"""
        print("正在为fairseq预处理数据...")
        
        # 创建预处理脚本
        preprocess_script = self.output_dir / "preprocess_custom.sh"
        
        script_content = f"""#!/bin/bash

# 预处理脚本
DATA_DIR="{self.training_data_dir}"
BART_DIR="MINDER/res/external/bart_large"  # 需要下载BART模型

# 使用fairseq预处理
fairseq-preprocess \
    --source-lang source \
    --target-lang target \
    --trainpref $DATA_DIR/train \
    --validpref $DATA_DIR/dev \
    --destdir $DATA_DIR/bin \
    --workers 20 \
    --srcdict $BART_DIR/dict.txt \
    --tgtdict $BART_DIR/dict.txt

echo "数据预处理完成"
"""
        
        with open(preprocess_script, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # 使脚本可执行
        os.chmod(preprocess_script, 0o755)
        
        print(f"预处理脚本已创建: {preprocess_script}")
        return preprocess_script
    
    def create_training_script(self):
        """创建训练脚本"""
        print("正在创建训练脚本...")
        
        training_script = self.output_dir / "train_custom_model.sh"
        
        script_content = f"""#!/bin/bash

# 训练脚本
DATA_DIR="{self.training_data_dir}/bin"
MODEL_DIR="{self.output_dir}/checkpoints"
BART_MODEL="MINDER/res/external/bart_large/model.pt"  # 需要下载BART模型

# 创建模型保存目录
mkdir -p $MODEL_DIR

# 开始训练
fairseq-train $DATA_DIR \
    --finetune-from-model $BART_MODEL \
    --arch bart_large \
    --task translation \
    --criterion label_smoothed_cross_entropy \
    --source-lang source \
    --target-lang target \
    --truncate-source \
    --label-smoothing 0.1 \
    --max-tokens 4096 \
    --update-freq 1 \
    --max-update 100000 \
    --required-batch-size-multiple 1 \
    --validate-interval 1000000 \
    --save-interval 1000000 \
    --save-interval-updates 5000 \
    --keep-interval-updates 3 \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --relu-dropout 0.0 \
    --weight-decay 0.01 \
    --optimizer adam \
    --adam-betas "(0.9, 0.999)" \
    --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay \
    --lr 3e-05 \
    --total-num-update 100000 \
    --warmup-updates 500 \
    --fp16 \
    --num-workers 10 \
    --no-epoch-checkpoints \
    --share-all-embeddings \
    --layernorm-embedding \
    --share-decoder-input-output-embed \
    --skip-invalid-size-inputs-valid-test \
    --log-format json \
    --log-interval 100 \
    --patience 3 \
    --find-unused-parameters \
    --save-dir $MODEL_DIR

echo "训练完成，模型保存在: $MODEL_DIR"
"""
        
        with open(training_script, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # 使脚本可执行
        os.chmod(training_script, 0o755)
        
        print(f"训练脚本已创建: {training_script}")
        return training_script
    
    def create_inference_script(self):
        """创建推理脚本"""
        print("正在创建推理脚本...")
        
        inference_script = self.output_dir / "inference_custom_model.sh"
        
        script_content = f"""#!/bin/bash

# 推理脚本
CHECKPOINT="{self.output_dir}/checkpoints/checkpoint_best.pt"
FM_INDEX="{self.fm_index_dir}/custom_corpus.fm_index"
TEST_QUERIES="{self.data_dir}/test_queries.csv"  # 需要创建测试查询文件
OUTPUT="{self.output_dir}/output_test.json"

# 运行推理
TOKENIZERS_PARALLELISM=false python MINDER/seal/search.py \
    --topics_format dpr_qas \
    --topics $TEST_QUERIES \
    --output_format dpr \
    --output $OUTPUT \
    --checkpoint $CHECKPOINT \
    --jobs 10 \
    --progress \
    --device cuda:0 \
    --batch_size 20 \
    --beam 15 \
    --decode_query stable \
    --fm_index $FM_INDEX

echo "推理完成，结果保存在: $OUTPUT"
"""
        
        with open(inference_script, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # 使脚本可执行
        os.chmod(inference_script, 0o755)
        
        print(f"推理脚本已创建: {inference_script}")
        return inference_script
    
    def create_setup_script(self):
        """创建环境设置脚本"""
        print("正在创建环境设置脚本...")
        
        setup_script = self.output_dir / "setup_environment.sh"
        
        script_content = """#!/bin/bash

# 环境设置脚本
echo "正在设置MINDER训练环境..."

# 检查MINDER项目
if [ ! -d "MINDER" ]; then
    echo "错误: 未找到MINDER项目目录"
    echo "请确保已克隆MINDER项目到当前目录"
    exit 1
fi

# 安装依赖
echo "安装系统依赖..."
sudo apt update
sudo apt install -y swig

# 编译sdsl-lite
echo "编译sdsl-lite..."
cd MINDER
env CFLAGS='-fPIC' CXXFLAGS='-fPIC' res/external/sdsl-lite/install.sh
cd ..

# 安装Python依赖
echo "安装Python依赖..."
pip install -r MINDER/requirements.txt
pip install -e MINDER/

# 下载BART模型 (需要手动下载)
echo "请手动下载BART-large模型到 MINDER/res/external/bart_large/ 目录"
echo "模型下载地址: https://github.com/pytorch/fairseq/blob/main/examples/bart/README.md"

echo "环境设置完成！"
echo "接下来请运行数据转换脚本"
"""
        
        with open(setup_script, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # 使脚本可执行
        os.chmod(setup_script, 0o755)
        
        print(f"环境设置脚本已创建: {setup_script}")
        return setup_script
    
    def run_full_pipeline(self):
        """运行完整的数据转换和准备流程"""
        print("开始运行完整的数据转换流程...")
        
        # 1. 加载数据
        data = self.load_data()
        
        # 2. 创建语料库文件
        corpus_file = self.create_corpus_file(data)
        
        # 3. 创建DPR格式数据
        train_file, dev_file = self.create_dpr_format_data(data)
        
        # 4. 创建伪查询映射
        pid2query_file = self.create_pseudo_queries_mapping(data)
        
        # 5. 生成监督训练数据
        self.generate_supervised_data(train_file, dev_file, pid2query_file)
        
        # 6. 生成无监督训练数据
        self.generate_unsupervised_data(corpus_file, pid2query_file)
        
        # 7. 合并训练数据
        self.merge_training_data()
        
        # 8. 构建FM-index
        self.build_fm_index(corpus_file)
        
        # 9. 创建各种脚本
        preprocess_script = self.preprocess_for_fairseq()
        training_script = self.create_training_script()
        inference_script = self.create_inference_script()
        setup_script = self.create_setup_script()
        
        print("\n=== 数据转换完成 ===")
        print(f"输出目录: {self.output_dir}")
        print("\n接下来的步骤:")
        print(f"1. 运行环境设置: bash {setup_script}")
        print(f"2. 预处理数据: bash {preprocess_script}")
        print(f"3. 开始训练: bash {training_script}")
        print(f"4. 运行推理: bash {inference_script}")
        print("\n注意: 请确保已下载BART-large模型到指定目录")

def main():
    parser = argparse.ArgumentParser(description="MINDER数据转换和训练脚本")
    parser.add_argument("input_json", help="输入的JSON数据文件路径")
    parser.add_argument("--output_dir", default="./minder_training", help="输出目录")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input_json):
        print(f"错误: 输入文件不存在: {args.input_json}")
        sys.exit(1)
    
    # 创建转换器并运行
    converter = MINDERDataConverter(args.input_json, args.output_dir)
    converter.run_full_pipeline()

if __name__ == "__main__":
    main()