bash scripts/download_model_dataset.sh

output_file_path=$(python3 main.py evaluation=random_in_context)
if [ -z "$output_file_path" ] || [ "$output_file_path" = "None" ]; then
    echo "Error: Failed to get output_file_path"
    exit 1
fi
echo "Results file: $output_file_path"

if [ -f "scripts/s3upload.sh" ]; then
    bash scripts/s3upload.sh ${output_file_path} ${output_file_path}
else
    echo "No s3upload.sh script found. Result will be saved locally."
fi
