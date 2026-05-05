from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.split_learning.trainer import train
from src.utils.seeding import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(__file__).resolve().parent.parent
    cfg.paths.project_root = str(project_root)
    seed_everything(int(cfg.seed), deterministic=bool(cfg.runtime.deterministic))
    print(OmegaConf.to_yaml(cfg))
    summary = train(cfg)
    print(summary)


if __name__ == "__main__":
    main()
