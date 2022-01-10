import jiant.scripts.download_data.runscript as downloader
import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.export_model as export_model
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.runscript as main_runscript
import jiant.utils.python.io as py_io

import sys
import os
import time


EXP_DIR = "./jiant"
BATCH_SIZE = 16
LEARNING_RATE = 3e-5
N_EPOCHS = 20

assert len(sys.argv) == 2
TASK = sys.argv[1]

is_diagnostics = TASK in ["superglue_broadcoverage_diagnostics", "superglue_winogender_diagnostics"]


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# Tokenize and cache the dataset
tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
    task_config_path=f"{EXP_DIR}/tasks/configs/{TASK}_config.json",
    hf_pretrained_model_name_or_path="en_bert",
    output_dir=f"{EXP_DIR}/cache/{TASK}",
    phases=["test"] if is_diagnostics else ["train", "val", "test"],
))

# Create the run config
jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
    task_config_base_path=f"{EXP_DIR}/tasks/configs",
    task_cache_base_path=f"{EXP_DIR}/cache",
    train_task_name_list=["rte" if is_diagnostics else TASK],
    val_task_name_list=["rte" if is_diagnostics else TASK],
    test_task_name_list=[TASK],
    train_batch_size=BATCH_SIZE,
    eval_batch_size=32,
    epochs=N_EPOCHS,
    num_gpus=1,
    warmup_steps_proportion=0.1,
).create_config()
os.makedirs(f"{EXP_DIR}/run_configs/", exist_ok=True)
py_io.write_json(jiant_run_config, f"{EXP_DIR}/run_configs/{TASK}_config.json")

for seed in [123, 456, 789, 1234, 5678, 9012, 12345, 67890, 123456, 789012]:
    output_dir = f"{EXP_DIR}/runs/run_{TASK}_seed={seed}"
    with HiddenPrints():
        run_args = main_runscript.RunConfiguration(
            jiant_task_container_config_path=f"{EXP_DIR}/run_configs/{TASK}_config.json",
            output_dir=output_dir,
            hf_pretrained_model_name_or_path="en_bert",
            model_path=f"{EXP_DIR}/models/en_bert/model/model.p",
            model_config_path=f"{EXP_DIR}/models/en_bert/model/config.json",
            learning_rate=LEARNING_RATE,
            adam_epsilon=1e-6,
            # eval_every_steps=0,
            do_train=True,
            do_val=True,
            # do_save=True,
            write_test_preds=True,
            force_overwrite=True,
            seed=seed,
        )
        start_time = time.time()
        main_runscript.run_loop(run_args)
        duration = time.time() - start_time

    with open(f"{output_dir}/time.txt", "w") as f:
        f.write(str(duration))
