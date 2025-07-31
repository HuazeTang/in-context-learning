bash scripts/download_model_dataset.sh

result=$(python3 main.py evaluation=infor_golden_in_context model=qwen | tee /dev/tty)
echo $result
output_file_path=$(echo $result | grep "Results saved to file:" | awk '{print $NF}')
if [ -z "$output_file_path" ] || [ "$output_file_path" = "None" ]; then
    echo "Error: Failed to get output_file_path"
    exit 1
fi
echo "$output_file_path"
# 转换为相对于当前目录的相对路径
relative_path=$(realpath --relative-to=. "$output_file_path")
echo "Results file (relative): $relative_path"

if [ -f "scripts/s3upload.sh" ]; then
    bash scripts/s3upload.sh ${relative_path} "qwen3-8b-mmlu-infor-golden-prompt"
else
    echo "No s3upload.sh script found. Result will be saved locally."
fi
