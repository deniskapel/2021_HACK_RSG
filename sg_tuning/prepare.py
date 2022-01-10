import jiant.scripts.download_data.runscript as downloader
import jiant.proj.main.export_model as export_model


EXP_DIR = "./jiant"

tasks = [
    "superglue_broadcoverage_diagnostics",  # Broadcoverage Diagnostics; Recognizing Textual Entailment
    "cb",  # CommitmentBank
    "copa",  # Choice of Plausible Alternatives
    "multirc",  # Multi-Sentence Reading Comprehension
    "wic",  # Words in Context
    "wsc",  # The Winograd Schema Challenge
    "boolq",  # BoolQ
    "record",  # Reading Comprehension with Commonsense Reasoning
    "superglue_winogender_diagnostics",  # Winogender Schema Diagnostics
    "rte"
]

# Download the Data
downloader.download_data(tasks, f"{EXP_DIR}/tasks")

# Cache the model
export_model.export_model(
    hf_pretrained_model_name_or_path="en_bert",
    output_base_path=f"{EXP_DIR}/models/en_bert",
)
