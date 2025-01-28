from torch.utils.data import Dataset, DataLoader

import os
from datasets import load_dataset, load_dataset_builder
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
import numpy as np
import pandas as pd
import mlflow

from utils import calculate_paired_similarities, compute_bleu_score

mlflow.set_tracking_uri("http://mlflow.local/")

os.environ["MLFLOW_EXPERIMENT_NAME"] = "ko-en-mt"

model_checkpoint = "Helsinki-NLP/opus-mt-ko-en"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")

# MinIO 스토리지 설정
MINIO_SERVER = "http://data.api.minio/"
ACCESS_KEY = "admin"  # MinIO access key
SECRET_KEY = "8KW4cWXuGc"  # MinIO secret key
# ACCESS_KEY = "api"
# SECRET_KEY = "gcH5KYnscWi7TDFZXGMoKNy00YArfsYZ1EqKuWAr"
# authenticate as built-in admin user



BUCKET_NAME = "ko-en-mt-tech"
MAX_LENGTH = 512

# 환경 변수로 MinIO 인증 정보 설정
os.environ["AWS_ACCESS_KEY_ID"] = ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = SECRET_KEY

# MinIO 설정
storage_options={
        "key": ACCESS_KEY,
        "secret": SECRET_KEY,
        "client_kwargs": {
            "endpoint_url": MINIO_SERVER,
        }
    }

S3_BUCKET = f"s3://ko-en-mt-tech/parquet"  # MinIO 버킷 및 파일 경로

np.random.seed(123)

dataset = load_dataset('parquet', data_files=S3_BUCKET + f"/train/*.parquet", storage_options=storage_options)
idx = np.arange(len(dataset['train']), dtype=int)
np.random.shuffle(idx)
idx = idx[:len(idx)//15]
dataset['train'] = dataset['train'].select(idx)

dataset['test'] = load_dataset('parquet', data_files=S3_BUCKET + f"/valid/*.parquet", storage_options=storage_options)['train']
idx = np.arange(len(dataset['test']), dtype=int)
np.random.shuffle(idx)
idx = idx[:len(idx)//100]
dataset['test'] = dataset['test'].select(idx)

def preprocessing_function(examples):    
    # print(examples)
    # inputs = [ex["ko"] for ex in examples["translation"]]
    # targets = [ex["en"] for ex in examples["translation"]]
    model_inputs = tokenizer(examples['ko'], text_target=examples['en'], max_length=MAX_LENGTH, truncation=True, padding=True,)
    return model_inputs

# 데이터셋 전처리
tk_dataset = dataset.map(preprocessing_function, 
                                batched=True, num_proc=16, 
                                # remove_columns=dataset["train"].column_names
                            )
                        

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
# DataCollator 준비 (lazy loading 지원)
# data_collator = DataCollatorForSeq2Seq(tokenizer, model=None, padding=True)

# # DataLoader 생성
# train_dataloader = DataLoader(
#     tk_train["train"],
#     batch_size=16,  # 배치 크기 설정
#     shuffle=True,
#     collate_fn=data_collator
# )

import evaluate

import numpy as np
from transformers import TrainerCallback

# class MLflowCallback(TrainerCallback):
#     def on_evaluate(self, args, state, control, metrics, **kwargs):
#         # Evaluate 단계에서 metric을 MLflow에 로깅
#         for key, value in metrics.items():
#             if key not in ["eval_loss", "eval_runtime", "eval_samples_per_second"]:
#                 mlflow.log_metric(key, value, step=state.global_step)
                
from transformers import Seq2SeqTrainingArguments

args = Seq2SeqTrainingArguments(
    
    f"marian-finetuned-kde4-ko-to-en",
    report_to="mlflow",
    # eval_strategy="steps",
    # eval_steps=50,
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    bf16=True,
    push_to_hub=False,
    eval_on_start=False,
    logging_steps=10,
    torch_compile=False,
    do_train=True,
    # max_steps=10
)

from transformers import Seq2SeqTrainer
from transformers import pipeline


def run_eval(model, name, tokenizer):
    tuned_pipeline = pipeline(
        task="translation",
        model=model,
        batch_size=32,
        tokenizer=tokenizer
    )
    
    # 3. 평가 데이터셋 예측
    to_slice = 100
    eval_set = tk_dataset['test']['ko'][:to_slice]
    truth = tk_dataset['test']['en'][:to_slice]
    predictions = [pred['translation_text'] for pred in tuned_pipeline(eval_set)]
    blue_score = compute_bleu_score(eval_set, truth, tuned_pipeline)
    cos_sim_dict, mat = calculate_paired_similarities(predictions, truth, tokenizer, model)
    cossim = [x['overall'] for x in cos_sim_dict]

    mlflow.log_metric(f"{name}-sacrebleu", blue_score['score'])
    mlflow.log_metric(f"{name}-cos", np.mean(cossim))
    
    # 4. 예측 결과 정리

    df = pd.DataFrame({
        "ko": eval_set,
        "translated": predictions,
        "en": truth,
        "cossim": cossim
    })

    mlflow.log_table(
        df, f"{name}-eval_result.json"
    )
    
# trainer.add_callback(MLflowCallback())
mlflow.transformers.autolog()

# Start MLflow run
with mlflow.start_run() as run:
    run_eval(model, name='init', tokenizer=tokenizer)
    
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tk_dataset["train"],
        eval_dataset=tk_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train the model
    trainer.train()
    components = {
        "model": trainer.model,
        "tokenizer": trainer.tokenizer,
    }
    # Log model
    mlflow.transformers.log_model(
        transformers_model=components,
        artifact_path="model",
        tokenizer=tokenizer,
        # input_example={"input_ids": np.array([0, 1, 2]), "attention_mask": np.array([1, 1, 1])},
        signature=None,
        extra_pip_requirements=["torch", "transformers", "datasets", "evaluate"],
        
    )
    run_eval(trainer.model, name='trained', tokenizer=trainer.tokenizer)
