"""Optional QLoRA fine-tuning of a small open-weight extractor (RQ2).

Data:   Data/annotations/qlora/{train,dev,test}.jsonl, one example per line:
        {"doc_id", "split", "text", "target": <ResumeExtraction|JobExtraction JSON>, "source": "teacher"|"human"}
        built by training/teacher_labels.py (teacher LLM output, optionally human-corrected).
        Splits follow Data/splits (near-duplicate clusters never cross splits).
Train:  python training/qlora.py train --base Qwen/Qwen2.5-1.5B-Instruct --out Data/indexes/qlora/adapter
Eval:   python training/qlora.py eval  --adapter Data/indexes/qlora/adapter
Needs:  CUDA GPU + bitsandbytes for 4-bit QLoRA. On Apple Silicon/CPU, 4-bit
        quantisation is unavailable; ``--no-4bit`` runs plain LoRA (slow).
The application never depends on this module.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import get_settings  # noqa: E402
from llm.prompts.extraction import JOB_SCHEMA_HINT, JOB_SYSTEM, RESUME_SCHEMA_HINT, RESUME_SYSTEM  # noqa: E402


def data_dir() -> Path:
    return get_settings().path("annotations") / "qlora"


def environment_status() -> dict:
    st = {"cuda": False, "bitsandbytes": False, "peft": False, "trl": False, "transformers": False}
    try:
        import torch

        st["cuda"] = torch.cuda.is_available()
        st["mps"] = torch.backends.mps.is_available()
    except Exception:
        pass
    for m in ("bitsandbytes", "peft", "trl", "transformers"):
        try:
            __import__(m)
            st[m] = True
        except Exception:
            pass
    files = {s: (data_dir() / f"{s}.jsonl").exists() for s in ("train", "dev", "test")}
    st["data"] = files
    st["can_train_qlora"] = st["cuda"] and st["bitsandbytes"] and st["peft"] and st["trl"] and all(files.values())
    st["blockers"] = [b for b, ok in (("CUDA GPU", st["cuda"]), ("bitsandbytes", st["bitsandbytes"]),
                                      ("peft", st["peft"]), ("trl", st["trl"]),
                                      ("training data (teacher labels)", all(files.values()))) if not ok]
    return st


def to_chat(example: dict) -> list[dict]:
    is_job = example.get("doc_type") == "job"
    system = JOB_SYSTEM if is_job else RESUME_SYSTEM
    hint = JOB_SCHEMA_HINT if is_job else RESUME_SCHEMA_HINT
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"{hint}\n\n<document>\n{example['text']}\n</document>"},
            {"role": "assistant", "content": json.dumps(example["target"], ensure_ascii=False)}]


def load_split(split: str) -> list[dict]:
    p = data_dir() / f"{split}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def train(base: str, out: str, use_4bit: bool = True, epochs: int = 2, lr: float = 2e-4, rank: int = 16) -> dict:
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    st = environment_status()
    if use_4bit and not st["can_train_qlora"]:
        raise RuntimeError(f"QLoRA training blocked: {st['blockers']}")
    tok = AutoTokenizer.from_pretrained(base)
    qconf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype="bfloat16") if use_4bit else None
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=qconf, device_map="auto")
    train_ds = Dataset.from_list([{"messages": to_chat(e)} for e in load_split("train")])
    dev_ds = Dataset.from_list([{"messages": to_chat(e)} for e in load_split("dev")])
    peft_cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(output_dir=out, num_train_epochs=epochs, learning_rate=lr, per_device_train_batch_size=1,
                    gradient_accumulation_steps=8, eval_strategy="epoch", save_strategy="epoch", logging_steps=10,
                    max_length=4096, bf16=use_4bit, report_to=[])
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=dev_ds, peft_config=peft_cfg,
                         processing_class=tok)
    trainer.train()
    trainer.save_model(out)
    meta = {"base": base, "adapter": out, "epochs": epochs, "lr": lr, "rank": rank, "4bit": use_4bit,
            "train_examples": len(train_ds), "dev_examples": len(dev_ds)}
    Path(out, "training_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_adapter(base: str, adapter: str):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
    return PeftModel.from_pretrained(model, adapter), tok


def status_result() -> dict:
    from evaluation.registry import write_result

    st = environment_status()
    return write_result("E_qlora", "blocked" if st["blockers"] else "not_evaluated",
                        reason=("Blocked: " + ", ".join(st["blockers"])) if st["blockers"] else
                               "Ready to train: run `python training/qlora.py train`.",
                        details={"environment": st})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    t = sub.add_parser("train")
    t.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    t.add_argument("--out", default=str(get_settings().path("indexes") / "qlora" / "adapter"))
    t.add_argument("--no-4bit", action="store_true")
    t.add_argument("--epochs", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "status":
        print(json.dumps(environment_status(), indent=2))
    elif a.cmd == "train":
        print(json.dumps(train(a.base, a.out, use_4bit=not a.no_4bit, epochs=a.epochs), indent=2))
