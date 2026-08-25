"""Model loading for the three configurations compared in the paper.

FP16          -- unquantized half precision
INT4          -- NF4 double quantization via bitsandbytes, FP16 compute
INT4+QLoRA    -- INT4 base with trained LoRA adapters attached
INT4+ZERO     -- control: INT4 base put through prepare_model_for_kbit_training
                 with a freshly initialised (zero-valued) adapter and NO
                 training. Isolates the dtype changes that preparation makes
                 from the effect of the adapter weights themselves.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

LORA_KWARGS = dict(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["qkv_proj", "o_proj"],
)


def load_tokenizer(model_id=MODEL_ID):
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_model(config, model_id=MODEL_ID, adapter_path=None, seed=42):
    """Return (model, tokenizer, weight_memory_gb) for the requested config."""
    torch.manual_seed(seed)
    tokenizer = load_tokenizer(model_id)

    common = dict(device_map={"": 0}, attn_implementation="eager")

    if config == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, **common)
    elif config in ("int4", "int4_qlora", "int4_zero"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, **common)
    else:
        raise ValueError(f"unknown config: {config}")

    weight_gb = model.get_memory_footprint() / 1e9

    if config == "int4_qlora":
        from peft import PeftModel
        if adapter_path is None:
            raise ValueError("int4_qlora requires --adapter-path")
        # Load adapters onto a FRESH base model. Do not reuse the in-memory
        # model that produced the INT4 numbers -- see README, "Clean ablation".
        model = PeftModel.from_pretrained(model, adapter_path)
    elif config == "int4_zero":
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(**LORA_KWARGS))

    model.config.use_cache = True
    model.eval()
    tokenizer.padding_side = "left"
    return model, tokenizer, weight_gb


def peak_memory_gb():
    """Peak allocated GPU memory since the last reset. Unlike
    get_memory_footprint() this includes activations and the KV cache, and is
    therefore the number that determines whether the model actually fits."""
    return torch.cuda.max_memory_allocated() / 1e9
