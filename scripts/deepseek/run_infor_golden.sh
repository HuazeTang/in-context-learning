# 设置变量
model="deepseek-qwen3-8b"
dataset="mmlu"
bash scripts/download_model_dataset.sh ${model} ${dataset}

method="infor-golden"
experiment_name="${model}_${dataset}_${method}"

# 启动评估任务（指定模型为 deepseek）
mkdir -p log
python3 main.py evaluation=infor_golden_in_context model=deepseek | tee "log/${experiment_name}.txt"

# 上传或保存结果
if [ -f "scripts/s3upload.sh" ]; then
    bash scripts/s3upload.sh "log/${experiment_name}"
else
    echo "No s3upload.sh script found. Result will be saved locally."
fi
