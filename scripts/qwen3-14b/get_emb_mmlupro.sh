# 设置变量
model="qwen_14b"
dataset="mmlu_pro" 
method="emb_gen"
pool_method="last" #mean 或 last 改变提取特征层位置
torch_dtype=float16

# 统一保存路径
save_dir="/root/autodl-tmp/embeddings/mmlupro"
mkdir -p "${save_dir}/results"
mkdir -p "${save_dir}/logs"

experiment_name="${model}_${output_name}_${method}_${pool_method}"

python3 main.py evaluation=emb_gen model=$model \
    model.config.torch_dtype=$torch_dtype \
    dataset=$dataset \
    evaluation.config.pool_method=${pool_method} \
    | tee "${save_dir}/logs/${experiment_name}.txt"

# 从日志中找 embeddings.pkl 的路径
result=$(cat "${save_dir}/logs/${experiment_name}.txt")
output_file_path=$(echo $result | grep "embeddings.pkl" | awk '{print $NF}')

if [ -n "$output_file_path" ] && [ "$output_file_path" != "None" ]; then
    new_filename="${output_name}_results.pkl"
    mv "$output_file_path" "${save_dir}/results/$new_filename"
    echo "Saved: ${save_dir}/results/$new_filename"
else
    echo "⚠️ 未找到结果文件，检查 ${save_dir}/logs/${experiment_name}.txt"
fi

# 打包结果
tar -czf "${save_dir}/${output_name}_results.tar.gz" -C "${save_dir}" results
echo "打包完成: ${save_dir}/${output_name}_results.tar.gz"
