import os
import shutil
import locale
import json
import logging
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict

import torch
from dotenv import load_dotenv

# Ensure UTF-8 locale for Unsloth
locale.getpreferredencoding = lambda *args, **kwargs: "UTF-8"

# Load .env if present
load_dotenv()

# Ensure C compiler for Triton / Unsloth
if "CC" not in os.environ:
    cc = shutil.which("gcc") or shutil.which("cc")
    if cc:
        os.environ["CC"] = cc

# Disable problematic Unsloth compilation cache
os.environ["UNSLOTH_COMPILE"] = "0"
os.environ["UNSLOTH_DISABLE_COMPILE"] = "1"
os.environ["UNSLOTH_DISABLE_DYNAMIC_COMPILE"] = "1"
os.environ["UNSLOTH_USE_TRITON"] = "0"

# MLflow System Metrics Tracking Setup
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

@dataclass
class TrainConfig:
    # Core model / data
    model_name: str = "google/gemma-3-1b-it"
    max_seq_length: int = 2048
    eval_seed: int = 42
    eval_samples: int = 100
    train_samples: int = 10_000

    # Paths
    output_dir: str = "outputs"
    save_dir: str = "gemma_1b_tool_call_lora"
    log_dir: str = "outputs"

    # Training hyperparameters
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    warmup_steps: int = 5
    max_steps: int | None = None  

    # Evaluation
    eval_show_samples: int = 10

    # Logging / tracking
    use_wandb: bool = False
    
    # Artifacts / registries
    wandb_model_name: str = "gemma-3-1b-tool-calling-lora"

    # Optional export targets
    push_to_hf: bool = False
    hf_repo_id: str | None = None
    push_to_minio: bool = False
    minio_endpoint: str | None = os.environ.get("MINIO_ENDPOINT")
    minio_access_key: str | None = os.environ.get("MINIO_ACCESS_KEY")
    minio_secret_key: str | None = os.environ.get("MINIO_SECRET_KEY")
    minio_bucket: str | None = os.environ.get("MINIO_BUCKET", "models")
    
    # MLflow tracking
    use_mlflow: bool = True
    mlflow_experiment_name: str = "gemma-3-tool-calling-finetuning"
    mlflow_tracking_uri: str = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow_s3_endpoint_url: str = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9100")

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def setup_logger():
    logger = logging.getLogger("gemma_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

logger = setup_logger()

# -------------------------------------------------------------
# Dataset and Prompts
# -------------------------------------------------------------
from datasets import load_dataset
from huggingface_hub import login
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

def process_example(example):
    return example

def load_tool_call_dataset(seed: int, test_size: int, train_size: int):
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    dataset = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    dataset = dataset.map(process_example, num_proc=4, desc="process answers")
    shuffled = dataset.shuffle(seed=seed)

    test_dataset = shuffled.select(range(test_size))
    train_dataset = shuffled.select(range(test_size, test_size + train_size))
    return train_dataset, test_dataset


def build_with_schema_hint_prompt(query: str, tool_definition: str) -> str:
    return f"""You are a tool-calling assistant.

Task:
Your task is to choose the appropriate tools to answer the user's query.

Instructions:
- Respond ONLY with the tools array in JSON format.

Available tools:
{tool_definition}

User query:
{query}
"""

def example_to_instruction_output(example):
    instruction = build_with_schema_hint_prompt(
        query=example["query"],
        tool_definition=example["tools"],
    )
    out = example["answers"]
    output = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
    return {"instruction": instruction, "output": output}

def formatting_prompts_func(examples, tokenizer):
    queries = examples["query"]
    tools_list = examples["tools"]
    answers_list = examples["answers"]

    convos = []
    for query, tools, answers in zip(queries, tools_list, answers_list):
        io = example_to_instruction_output({"query": query, "tools": tools, "answers": answers})
        instruction = io["instruction"]
        output = io["output"]

        convos.append(
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": instruction}],
                },
                {
                    "role": "model",
                    "content": [{"type": "text", "text": output}],
                },
            ]
        )

    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False,
        ).removeprefix("<bos>")
        for convo in convos
    ]
    return {"text": texts}

# -------------------------------------------------------------
# Model Loading
# -------------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback


class BestLossCallback(TrainerCallback):
    def __init__(self, save_dir, tokenizer, use_mlflow=False):
        self.best_loss = float("inf")
        self.best_step = 0
        self.save_dir = save_dir
        self.tokenizer = tokenizer
        self.use_mlflow = use_mlflow

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        
        # Explicitly flush training logs to MLflow to ensure they appear in the UI immediately
        # if self.use_mlflow:
        #     import mlflow
        #     if mlflow.active_run():
        #         numeric_logs = {k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}
        #         mlflow.log_metrics(numeric_logs, step=state.global_step)


        loss = logs["loss"]
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_step = state.global_step
            logger.info("Step %d: new best loss %.4f — saving adapter to %s", state.global_step, loss, self.save_dir)
            model.save_pretrained(self.save_dir)
            self.tokenizer.save_pretrained(self.save_dir)


def get_model_and_tokenizer(cfg: TrainConfig):
    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name=cfg.model_name,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=False,
            load_in_8bit=False,
            full_finetuning=False,
            dtype=None,
        )
    except Exception as e:
        logger.error("Error loading model with Unsloth: %s", e)
        logger.info("Falling back to transformers + PEFT.")
        from peft import LoraConfig, get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return model, tokenizer

    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=8,
        lora_alpha=8,
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )
    return model, tokenizer

# -------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------
def build_prompt_for_eval(example):
    return example_to_instruction_output(example)["instruction"]

def parse_model_output(text: str):
    text = (text or "").strip()
    if not text:
        return None

    for token in ["<end_of_turn>", "<start_of_turn>", "<bos>", "<eos>"]:
        text = text.replace(token, "")
    text = text.strip()

    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part and (part.startswith("[") or part.startswith("{")):
                text = part
                break

    for prefix in ("model\n", "model ", "user\n", "user "):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    for stop in ["###", "<|", "\n\n\n"]:
        if stop in text:
            text = text.split(stop)[0].strip()

    if "[" in text:
        start = text.find("[")
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : i + 1])
                        return parsed if isinstance(parsed, list) else None
                    except json.JSONDecodeError:
                        pass

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    return None

def evaluate_tool_calls(generated, ground_truth):
    return generated == ground_truth

def run_eval(model, tokenizer, test_dataset, device="cuda", show_samples: int = 3, batch_size: int = 16):
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    except Exception:
        model.eval()
        
    exact_matches = 0
    total = len(test_dataset)
    logger.info("Running eval on %d samples...", total)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    for i in range(0, total, batch_size):
        batch_exs = [test_dataset[j] for j in range(i, min(i + batch_size, total))]
        
        messages_batch = []
        gts = []
        for ex in batch_exs:
            messages_batch.append([
                {
                    "role": "user",
                    "content": [{"type": "text", "text": build_prompt_for_eval(ex)}],
                }
            ])
            try:
                gts.append(json.loads(ex["answers"]))
            except (TypeError, json.JSONDecodeError):
                gts.append([])

        inputs = tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
        ).to(device)

        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
            )

        new_tokens = outputs[:, input_len:]
        responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        
        for k, response in enumerate(responses):
            idx = i + k
            response = response.strip()
            generated = parse_model_output(response)
            gt = gts[k]

            is_match = generated is not None and evaluate_tool_calls(generated, gt)
            if is_match:
                exact_matches += 1

            if idx < show_samples:
                logger.debug("Sample %d/%d - Match: %s", idx + 1, total, "✓" if is_match else "✗")

    rate = (exact_matches / total * 100) if total else 0.0
    logger.info("Eval complete: %d/%d exact matches (%.2f%%)", exact_matches, total, rate)
    return rate

# -------------------------------------------------------------
# Main Loop
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()

    cfg = TrainConfig.from_json(args.config) if os.path.exists(args.config) else TrainConfig()
    logger.info("Config: %s", json.dumps(asdict(cfg), indent=2, default=str))

    if not torch.cuda.is_available():
        logger.warning("CUDA GPU not detected.")

    import mlflow
    if cfg.use_mlflow:
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("AWS_ACCESS_KEY_ID", cfg.minio_access_key or "minioadmin")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("AWS_SECRET_ACCESS_KEY", cfg.minio_secret_key or "minioadmin123")
        os.environ["AWS_DEFAULT_REGION"] = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = cfg.mlflow_s3_endpoint_url

        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.mlflow_experiment_name)
        mlflow_run = mlflow.start_run()
        logger.info("Initialized MLflow run ID: %s", mlflow_run.info.run_id)
        
        try:
            mlflow.enable_system_metrics_logging()
            logger.info("Explicitly enabled MLflow system metrics logging.")
        except Exception as e:
            logger.warning("Could not manually enable system metrics: %s", e)
            
        mlflow.log_params(asdict(cfg))
        
        config_dict = asdict(cfg)
        with open("train_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        mlflow.log_artifact("train_config.json", "setup_artifacts")
    else:
        logger.info("cfg.use_mlflow = False -> skipping MLflow initialization.")

    train_raw, test_dataset = load_tool_call_dataset(
        seed=cfg.eval_seed, test_size=cfg.eval_samples, train_size=cfg.train_samples
    )

    model, tokenizer = get_model_and_tokenizer(cfg)

    train_dataset = train_raw.map(
        lambda batch: formatting_prompts_func(batch, tokenizer),
        batched=True,
    )

    if cfg.use_mlflow:
        logger.info("Saving training dataset locally for MLflow logging...")
        train_dataset.to_json("train_dataset.jsonl", orient="records", lines=True)
        mlflow.log_artifact("train_dataset.jsonl", "data")

    # Eval pre
    # initial_accuracy = run_eval(model, tokenizer, test_dataset, show_samples=cfg.eval_show_samples)
    initial_accuracy = 0.0
    if cfg.use_mlflow:
        mlflow.log_metric("eval/pre_train_accuracy", initial_accuracy)

    best_loss_cb = BestLossCallback(save_dir=cfg.save_dir, tokenizer=tokenizer, use_mlflow=cfg.use_mlflow)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            warmup_steps=cfg.warmup_steps,
            max_steps=cfg.max_steps if cfg.max_steps is not None else -1,
            learning_rate=cfg.learning_rate,
            logging_steps=1,
            num_train_epochs=cfg.num_train_epochs,
            optim="adamw_8bit",
            weight_decay=cfg.weight_decay,
            lr_scheduler_type="linear",
            seed=cfg.eval_seed,
            output_dir=cfg.output_dir,
            report_to=["mlflow"] if cfg.use_mlflow else [],
            save_strategy="no",
        ),
        callbacks=[best_loss_cb],
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    logger.info("Starting training...")
    trainer.train()

    # Eval post
    best_model, best_tokenizer = FastModel.from_pretrained(
        model_name=cfg.save_dir, max_seq_length=cfg.max_seq_length, load_in_4bit=False, dtype=None
    )
    best_tokenizer = get_chat_template(best_tokenizer, chat_template="gemma-3")
    # final_accuracy = run_eval(best_model, best_tokenizer, test_dataset, show_samples=cfg.eval_show_samples)
    final_accuracy = 0.0

    if cfg.use_mlflow:
        mlflow.log_metric("eval/post_train_accuracy", final_accuracy)
        mlflow.log_metric("eval/improvement", final_accuracy - initial_accuracy)
        logger.info("Logging adapter model to MLflow artifact registry -> s3://%s/mlflow", cfg.minio_bucket)
        mlflow.log_artifacts(cfg.save_dir, artifact_path="model")
        mlflow.end_run()

if __name__ == "__main__":
    main()
