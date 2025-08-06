#!/bin/bash

# MINDER 一键训练脚本
# 使用方法: bash one_click_train.sh <input_json_file> [output_dir]
# 示例: bash one_click_train.sh sample_data.json ./minder_training

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查参数
if [ $# -lt 1 ]; then
    log_error "使用方法: bash $0 <input_json_file> [output_dir]"
    log_info "示例: bash $0 sample_data.json ./minder_training"
    exit 1
fi

INPUT_JSON="$1"
OUTPUT_DIR="${2:-./minder_training}"

# 检查输入文件
if [ ! -f "$INPUT_JSON" ]; then
    log_error "输入文件不存在: $INPUT_JSON"
    exit 1
fi

log_info "开始MINDER一键训练流程"
log_info "输入文件: $INPUT_JSON"
log_info "输出目录: $OUTPUT_DIR"

# 记录开始时间
START_TIME=$(date +%s)

# 步骤1: 检查和准备环境
log_info "步骤1/8: 检查和准备环境"

# 检查MINDER项目
if [ ! -d "MINDER" ]; then
    log_warning "未找到MINDER项目，正在克隆..."
    git clone https://github.com/microsoft/MINDER.git
    if [ $? -ne 0 ]; then
        log_error "克隆MINDER项目失败"
        exit 1
    fi
fi

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    log_error "未找到python3，请先安装Python 3.7+"
    exit 1
fi

# 检查CUDA
if ! command -v nvidia-smi &> /dev/null; then
    log_warning "未检测到NVIDIA GPU，训练可能会很慢"
else
    log_info "检测到NVIDIA GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
fi

# 安装系统依赖
log_info "安装系统依赖..."
sudo apt update -qq
sudo apt install -y swig build-essential

# 编译sdsl-lite
log_info "编译sdsl-lite库..."
cd MINDER
if [ ! -f "res/external/sdsl-lite/lib/libsdsl.a" ]; then
    env CFLAGS='-fPIC' CXXFLAGS='-fPIC' res/external/sdsl-lite/install.sh
else
    log_info "sdsl-lite已编译，跳过"
fi
cd ..

# 安装Python依赖
log_info "安装Python依赖..."
pip install -q -r MINDER/requirements.txt
pip install -q -e MINDER/

# 安装额外依赖
pip install -q tqdm fuzzywuzzy python-Levenshtein

log_success "环境准备完成"

# 步骤2: 检查和下载BART模型
log_info "步骤2/8: 检查BART模型"

BART_DIR="MINDER/res/external/bart_large"
if [ ! -f "$BART_DIR/model.pt" ] || [ ! -f "$BART_DIR/dict.txt" ]; then
    log_info "BART模型不存在，正在下载..."
    mkdir -p "$BART_DIR"
    
    # 下载BART模型
    cd "$BART_DIR"
    
    # 下载模型文件
    if [ ! -f "model.pt" ]; then
        log_info "下载BART模型文件..."
        wget -q https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz
        tar -xzf bart.large.tar.gz
        mv bart.large/model.pt .
        mv bart.large/dict.txt .
        rm -rf bart.large bart.large.tar.gz
    fi
    
    cd ../../..
else
    log_info "BART模型已存在，跳过下载"
fi

log_success "BART模型准备完成"

# 步骤3: 数据转换
log_info "步骤3/8: 数据转换"

python3 data_converter_and_trainer.py "$INPUT_JSON" --output_dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    log_error "数据转换失败"
    exit 1
fi

log_success "数据转换完成"

# 进入输出目录
cd "$OUTPUT_DIR"

# 步骤4: 生成监督训练数据
log_info "步骤4/8: 生成监督训练数据"

# 生成三种视角的训练数据
for split in train dev; do
    input_file="data/custom-${split}.json"
    
    for target in title span query; do
        log_info "生成 ${split} 集的 ${target} 数据..."
        
        case $target in
            "title")
                n_samples=3
                mode="w"
                ;;
            "span")
                n_samples=10
                mode="a"
                ;;
            "query")
                n_samples=5
                mode="a"
                ;;
        esac
        
        cmd="python3 ../MINDER/scripts/training/make_supervised_dpr_dataset.py \
            $input_file \
            data/training_data/custom_dataset/${split} \
            --target $target \
            --mark_target \
            --mark_silver \
            --n_samples $n_samples \
            --mode $mode \
            --min_score 0.0 \
            --min_score_gold 0.0"
        
        if [ "$target" = "query" ]; then
            cmd="$cmd --pid2query data/pseudo_queries/pid2query_custom.pkl"
        fi
        
        eval $cmd
        
        if [ $? -ne 0 ]; then
            log_error "生成 ${target} 数据失败"
            exit 1
        fi
    done
done

log_success "监督训练数据生成完成"

# 步骤5: 生成无监督训练数据
log_info "步骤5/8: 生成无监督训练数据"

python3 ../MINDER/scripts/training/make_generated_dataset2.py \
    data/custom_corpus.tsv \
    data/training_data/custom_dataset/unsupervised.source \
    data/training_data/custom_dataset/unsupervised.target \
    --format dpr \
    --num_samples 3 \
    --num_title_samples 1 \
    --num_query_samples 2 \
    --full_doc_n 1 \
    --mark_pretraining \
    --pid2query data/pseudo_queries/pid2query_custom.pkl

if [ $? -ne 0 ]; then
    log_error "生成无监督数据失败"
    exit 1
fi

# 合并训练数据
log_info "合并训练数据..."
cat data/training_data/custom_dataset/unsupervised.source >> data/training_data/custom_dataset/train.source
cat data/training_data/custom_dataset/unsupervised.target >> data/training_data/custom_dataset/train.target

log_success "无监督训练数据生成完成"

# 步骤6: 构建FM-index
log_info "步骤6/8: 构建FM-index"

python3 ../MINDER/scripts/build_fm_index.py \
    data/custom_corpus.tsv \
    data/fm_index/custom/custom_corpus.fm_index

if [ $? -ne 0 ]; then
    log_error "构建FM-index失败"
    exit 1
fi

log_success "FM-index构建完成"

# 步骤7: 预处理数据
log_info "步骤7/8: 预处理数据"

DATA_DIR="data/training_data/custom_dataset"
BART_DIR="../MINDER/res/external/bart_large"

# 检查数据文件
if [ ! -f "$DATA_DIR/train.source" ] || [ ! -f "$DATA_DIR/train.target" ]; then
    log_error "训练数据文件不存在"
    exit 1
fi

# 使用fairseq预处理
log_info "使用fairseq预处理数据..."
fairseq-preprocess \
    --source-lang source \
    --target-lang target \
    --trainpref $DATA_DIR/train \
    --validpref $DATA_DIR/dev \
    --destdir $DATA_DIR/bin \
    --workers 20 \
    --srcdict $BART_DIR/dict.txt \
    --tgtdict $BART_DIR/dict.txt

if [ $? -ne 0 ]; then
    log_error "数据预处理失败"
    exit 1
fi

log_success "数据预处理完成"

# 步骤8: 开始训练
log_info "步骤8/8: 开始模型训练"

DATA_BIN_DIR="$DATA_DIR/bin"
MODEL_DIR="checkpoints"
BART_MODEL="$BART_DIR/model.pt"

# 创建模型保存目录
mkdir -p $MODEL_DIR

# 检查GPU内存并调整批次大小
if command -v nvidia-smi &> /dev/null; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ $GPU_MEM -lt 16000 ]; then
        MAX_TOKENS=2048
        log_warning "GPU内存较小，调整批次大小为 $MAX_TOKENS"
    else
        MAX_TOKENS=4096
    fi
else
    MAX_TOKENS=1024
    log_warning "未检测到GPU，使用CPU训练，批次大小调整为 $MAX_TOKENS"
fi

# 开始训练
log_info "开始训练模型，这可能需要几个小时..."
log_info "训练参数: max-tokens=$MAX_TOKENS, max-update=100000"

fairseq-train $DATA_BIN_DIR \
    --finetune-from-model $BART_MODEL \
    --arch bart_large \
    --task translation \
    --criterion label_smoothed_cross_entropy \
    --source-lang source \
    --target-lang target \
    --truncate-source \
    --label-smoothing 0.1 \
    --max-tokens $MAX_TOKENS \
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
    --save-dir $MODEL_DIR 2>&1 | tee $MODEL_DIR/train.log

if [ $? -ne 0 ]; then
    log_error "模型训练失败，请检查日志: $MODEL_DIR/train.log"
    exit 1
fi

log_success "模型训练完成"

# 计算总耗时
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

log_success "=== 训练流程全部完成 ==="
log_info "总耗时: ${HOURS}小时${MINUTES}分钟${SECONDS}秒"
log_info "模型保存位置: $(pwd)/$MODEL_DIR"
log_info "训练日志: $(pwd)/$MODEL_DIR/train.log"
log_info "FM-index: $(pwd)/data/fm_index/custom/custom_corpus.fm_index"

# 创建测试脚本
log_info "创建推理测试脚本..."
cat > test_model.sh << 'EOF'
#!/bin/bash

# 模型推理测试脚本
# 使用方法: bash test_model.sh "测试查询"

if [ $# -lt 1 ]; then
    echo "使用方法: bash $0 \"测试查询\""
    echo "示例: bash $0 \"什么是人工智能？\""
    exit 1
fi

QUERY="$1"
CHECKPOINT="checkpoints/checkpoint_best.pt"
FM_INDEX="data/fm_index/custom/custom_corpus.fm_index"
OUTPUT="test_output.json"

# 创建临时查询文件
echo "query_id,query" > temp_query.csv
echo "1,$QUERY" >> temp_query.csv

# 运行推理
TOKENIZERS_PARALLELISM=false python ../MINDER/seal/search.py \
    --topics_format dpr_qas \
    --topics temp_query.csv \
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
echo "查询: $QUERY"
echo "结果:"
cat $OUTPUT | jq '.'

# 清理临时文件
rm temp_query.csv
EOF

chmod +x test_model.sh

log_success "推理测试脚本已创建: test_model.sh"
log_info "测试模型: bash test_model.sh \"你的查询\""

echo
log_success "🎉 MINDER模型训练完成！"
echo
log_info "接下来你可以:"
log_info "1. 测试模型: bash test_model.sh \"测试查询\""
log_info "2. 查看训练日志: tail -f $MODEL_DIR/train.log"
log_info "3. 使用模型进行批量推理"
echo