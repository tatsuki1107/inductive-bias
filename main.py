from pathlib import Path
import joblib
import hydra
from omegaconf import DictConfig
from utils.dataloader import synthesize_data
from feedbackloop.random import random_loop
from feedbackloop.mf import adamMF_loop
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    config = cfg.settings
    algo_list = config.algo
    data_path = Path("./data/synthetic.pkl")

    if not config.dataset.is_created:
        data, rate_time = synthesize_data(
            dataparams=config.dataset,
            seed=config.random_seed,
        )
        with open(data_path, "wb") as f:
            joblib.dump((data, rate_time), f)

    with open(data_path, "rb") as f:
        data, rate_time = joblib.load(f)

    result_df = pd.DataFrame(
        columns=[
            "algorithm",
            "jaccard_index",
            "gini_coef",
            "train_loss",
            "test_loss",
        ]
    )

    for algo in tqdm(algo_list):
        if algo == "random":
            random_ginis, random_jaccrds, random_matrix = random_loop(
                data=data,
                rate_time=rate_time,
                feedback_epochs=config.feedback_epochs,
                rating_scale=config.dataset.rating_scale,
                addnum=config.addnum,
                seed=config.random_seed,
            )
            _df = pd.DataFrame(
                {
                    "algorithm": [algo] * config.feedback_epochs,
                    "jaccard_index": random_jaccrds,
                    "gini_coef": random_ginis,
                    "train_loss": [None] * config.feedback_epochs,
                    "test_loss": [None] * config.feedback_epochs,
                }
            )
            result_df = pd.concat([result_df, _df], ignore_index=True)

        elif algo == "conventional_mf":
            (
                conv_train_losses,
                conv_test_losses,
                conv_ginis,
                conv_jaccards,
                conv_matrix,
            ) = adamMF_loop(
                data=data,
                rate_time=rate_time,
                feedback_epochs=config.feedback_epochs,
                addnum=config.addnum,
                seed=config.random_seed,
                model_params=config.mf,
            )
            _df = pd.DataFrame(
                {
                    "algorithm": [algo] * config.feedback_epochs,
                    "jaccard_index": conv_jaccards,
                    "gini_coef": conv_ginis,
                    "train_loss": conv_train_losses,
                    "test_loss": conv_test_losses,
                }
            )
            result_df = pd.concat([result_df, _df], ignore_index=True)

        elif algo == "ips_mf":
            (
                ips_train_losses,
                ips_test_losses,
                ips_ginis,
                ips_jaccards,
                ips_matrix,
            ) = adamMF_loop(
                data=data,
                rate_time=rate_time,
                feedback_epochs=config.feedback_epochs,
                addnum=config.addnum,
                seed=config.random_seed,
                model_params=config.mf,
                universal=True,
            )
            _df = pd.DataFrame(
                {
                    "algorithm": [algo] * config.feedback_epochs,
                    "jaccard_index": ips_jaccards,
                    "gini_coef": ips_ginis,
                    "train_loss": ips_train_losses,
                    "test_loss": ips_test_losses,
                }
            )
            result_df = pd.concat([result_df, _df], ignore_index=True)

    log_path = Path(f"{HydraConfig.get().runtime.output_dir}/results")
    log_path.mkdir(exist_ok=True, parents=True)

    result_df.to_csv(log_path / "result.csv", index=False)

    with open(log_path / "log_data.pkl", "wb") as f:
        joblib.dump([conv_matrix, ips_matrix, random_matrix], f)


if __name__ == "__main__":
    main()
