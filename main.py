import os
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from hydra.utils import instantiate
from src.models import BaseModel
from src.datasets import BaseDataset
from src.evaluators import BaseEvaluator


# 设置日志
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="./configs", config_name="main")
def main(cfg: DictConfig):
    # 打印配置
    log.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    
    # 加载模型
    log.info(f"Loading {cfg.model.config.type} model...")
    model: BaseModel = instantiate(cfg.model)
    
    # 实例化数据集
    log.info(f"Loading {cfg.dataset.config.name} dataset")
    dataset: BaseDataset = instantiate(cfg.dataset)
    
    # 实例化评估器
    log.info(f"Loading {cfg.evaluation.config.name} evaluator")
    evaluator: BaseEvaluator = instantiate(cfg.evaluation, model=model, dataset=dataset, logger=log)
    
    # 运行评估
    log.info("\nStarting evaluation...")
    accuracy, results, output_path = evaluator.evaluate(output_dir=cfg.evaluation.config.output_dir)

    # 输出结果    
    log.info(f"\nEvaluation completed. Accuracy: {accuracy:.4f}")

    # 输出结果
    log.info(f"Results saved to {output_path}")

    return output_path


if __name__ == "__main__":
    output_file_path = main()
    print(output_file_path, flush=True)