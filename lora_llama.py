import torch
import torch.nn as nn
from transformers import AutoModel

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

try:
    from peft import LoraConfig, get_peft_model, TaskType
except Exception:
    LoraConfig = None
    get_peft_model = None
    TaskType = None


class DecoderClassification(nn.Module):
    """
    Minimal classifier wrapper for decoder-only backbones loaded via AutoModel.
    Uses last_hidden_state + masked mean pooling + linear head.

    IMPORTANT:
    - We keep classifier on the same device as the pooled features to avoid cpu/cuda mismatch.
    """
    def __init__(self, backbone, num_labels=2):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None, return_dict=True, **kwargs):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs
        )
        x = out.last_hidden_state  # [B,T,H]

        if attention_mask is None:
            pooled = x.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(x.dtype)
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        # Safety: ensure pooled and classifier are on same device
        pooled = pooled.to(device=self.classifier.weight.device,
                   dtype=self.classifier.weight.dtype)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        if return_dict:
            return type("Out", (), {"loss": loss, "logits": logits, "hidden_states": getattr(out, "hidden_states", None)})
        return (loss, logits)


def _make_quant_config(args):
    if not (getattr(args, "load_in_4bit", False) or getattr(args, "load_in_8bit", False)):
        return None, None
    if BitsAndBytesConfig is None:
        raise ImportError(
            "BitsAndBytesConfig not available. Upgrade transformers or disable --load_in_4bit/--load_in_8bit."
        )
    device_map = "auto"
    if getattr(args, "load_in_4bit", False):
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16 if getattr(args, "fp16", False) else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        return quant_cfg, device_map
    if getattr(args, "load_in_8bit", False):
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        return quant_cfg, device_map
    return None, None


def _apply_lora(backbone, args):
    if not getattr(args, "use_lora", False):
        return backbone
    if get_peft_model is None:
        raise ImportError("peft not installed. pip install peft")
    # Common projection module suffixes for LLaMA/Qwen/Mistral HF implementations
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(
        r=getattr(args, "lora_r", 8),
        lora_alpha=getattr(args, "lora_alpha", 16),
        lora_dropout=getattr(args, "lora_dropout", 0.05),
        bias="none",
        target_modules=target_modules,
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    backbone = get_peft_model(backbone, lora_cfg)
    try:
        backbone.print_trainable_parameters()
    except Exception:
        pass
    return backbone


def build_llama_with_lora(model_name: str, num_labels: int, args):
    """LLaMA backbone + optional quantization + optional LoRA."""
    quant_cfg, device_map = _make_quant_config(args)

    backbone = AutoModel.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        device_map=device_map,
        torch_dtype=torch.float16 if getattr(args, "fp16", False) else None,
    )

    if getattr(args, "grad_checkpoint", False) and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()

    backbone = _apply_lora(backbone, args)

    model = DecoderClassification(backbone, num_labels=num_labels)

    # Ensure head is on same device as backbone (fixes cpu/cuda mismatch at classifier)
    try:
        head_device = next(p.device for p in backbone.parameters() if p is not None)
        model.classifier = model.classifier.to(head_device)
    except StopIteration:
        pass

    return model
