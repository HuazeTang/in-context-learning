#model_name=$1
#dataset_name=$2
#
#if [ -f "scripts/download_model_dataset_hope.sh" ]; then
#    echo "Download with hope"
#    bash scripts/download_model_dataset_hope.sh ${model_name} ${dataset_name}
#else   
#    mkdir -p data/datasets
#    mkdir -p data/model_ckpts
#
#    if [ "$model_name" = "llama3-8b" ]; then 
#        if [ ! -d "data/model_ckpts/meta-llama/Meta-Llama-3-8B-Instruct" ]; then
#            mkdir -p data/model_ckpts/meta-llama
#            git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct data/model_ckpts/meta-llama/Meta-Llama-3-8B-Instruct
#        else
#            echo "Meta-Llama-3-8B-Instruct already exists"
#        fi
#    elif [ "$model_name" = "qwen3-8b" ]; then
#        if [ ! -d "data/model_ckpts/Qwen/Qwen3-8B" ]; then
#            mkdir -p data/model_ckpts/Qwen
#            git clone https://huggingface.co/Qwen/Qwen3-8B data/model_ckpts/Qwen/Qwen3-8B data/model_ckpts/Qwen/Qwen3-8B
#        else
#            echo "Qwen3-8B already exists"
#        fi
#    else
#        echo "Unknown model name: $model_name"
#        echo "Supported models: llama3-b, qwen3-8b"
#        exit 1
#    fi
#
#    if [ "$dataset_name" = "mmlu" ]; then
#        if [ ! -d "data/datasets/cais/mmlu" ]; then
#            mkdir -p data/datasets/cais/mmlu
#            git clone https://huggingface.co/datasets/cais/mmlu data/datasets/cais/mmlu
#        else
#            echo "mmlu already exists"
#        fi
#    else
#        echo "Unknown dataset name: $dataset_name"
#        echo "Supported dataset: mmlu"
#        exit 1
#    fi
#fi



#if [ -f "scripts/download_model_dataset_hope.sh" ]; then
#    echo "Download with hope"
#    bash scripts/download_model_dataset_hope.sh
#else
#    mkdir -p /root/autodl-tmp/data/datasets
#    mkdir -p /root/autodl-tmp/data/model_ckpts
#
#    if [ ! -d "/root/autodl-tmp/data/model_ckpts/meta-llama/Meta-Llama-3-8B-Instruct" ]; then
#        mkdir -p /root/autodl-tmp/data/model_ckpts/meta-llama
#        git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct data/model_ckpts/meta-llama/Meta-Llama-3-8B-Instruct
#    else
#        echo "Meta-Llama-3-8B-Instruct already exists"
#    fi
#
#    if [ ! -d "/root/autodl-tmp/data/model_ckpts/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B" ]; then
#        mkdir -p data/model_ckpts/deepseek-ai
#        git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B data/model_ckpts/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
#    else
#        echo "DeepSeek-R1-0528-Qwen3-8B already exists"
#    fi
#
#    if [ ! -d "/root/autodl-tmp/data/model_ckpts/Qwen/Qwen3-8B" ]; then
#        mkdir -p /root/autodl-tmp/data/model_ckpts/Qwen
#        git clone https://huggingface.co/Qwen/Qwen3-8B data/model_ckpts/Qwen/Qwen3-8B
#    else
#        echo "Qwen3-8B already exists"
#    fi
#
#    if [ ! -d "/root/autodl-tmp/data/datasets/cais/mmlu" ]; then
#        mkdir -p /root/autodl-tmp/data/datasets/cais/mmlu
#        git clone https://huggingface.co/datasets/cais/mmlu data/datasets/cais/mmlu
#    else
#        echo "mmlu already exists"
#    fi
#fi

if [ -f "scripts/download_model_dataset_hope.sh" ]; then
    echo "Download with hope"
    bash scripts/download_model_dataset_hope.sh
else
    mkdir -p /root/autodl-tmp/data/datasets
    mkdir -p /root/autodl-tmp/data/model_ckpts

    if [ ! -d "/root/autodl-tmp/data/model_ckpts/meta-llama/Meta-Llama-3-8B-Instruct" ]; then
        mkdir -p /root/autodl-tmp/data/model_ckpts/meta-llama
        git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct data/model_ckpts/meta-llama/Meta-Llama-3-8B-Instruct
    else
        echo "Meta-Llama-3-8B-Instruct already exists"
    fi

    if [ ! -d "/root/autodl-tmp/data/model_ckpts/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B" ]; then
        mkdir -p data/model_ckpts/deepseek-ai
        git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B data/model_ckpts/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
    else
        echo "DeepSeek-R1-0528-Qwen3-8B already exists"
    fi

    if [ ! -d "/root/autodl-tmp/data/model_ckpts/Qwen/Qwen3-14B" ]; then
        mkdir -p /root/autodl-tmp/data/model_ckpts/Qwen
        git clone https://huggingface.co/Qwen/Qwen3-14B data/model_ckpts/Qwen/Qwen3-14B
    else
        echo "Qwen3-8B already exists"
    fi

    if [ ! -d "/root/autodl-tmp/data/datasets/cais/mmlu" ]; then
        mkdir -p /root/autodl-tmp/data/datasets/cais/mmlu
        git clone https://huggingface.co/datasets/cais/mmlu data/datasets/cais/mmlu
    else
        echo "mmlu already exists"
    fi

    if [ ! -d "/root/autodl-tmp/data/datasets/TIGER-Lab/MMLU-Pro" ]; then
        mkdir -p /root/autodl-tmp/data/datasets/TIGER-Lab/MMLU-Pro
        git clone https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro data/datasets/TIGER-Lab/MMLU-Pro
    else
        echo "MMLU-Pro already exists"
    fi

    if [ ! -d "/root/autodl-tmp/data/datasets/Idavidrein/gpqa" ]; then
        mkdir -p /root/autodl-tmp/data/datasets/Idavidrein/gpqa
        git clone https://huggingface.co/datasets/Idavidrein/gpqa data/datasets/Idavidrein/gpqa
    else
        echo "gpqa already exists"
    fi
fi