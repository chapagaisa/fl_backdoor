import torch
import torch.nn as nn
from transformers import AutoModel

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
except Exception:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
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
        pooled = pooled.to(device=self.classifier.weight.device,dtype=self.classifier.weight.dtype)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        if return_dict:
            return type("Out", (), {"loss": loss, "logits": logits, "hidden_states": getattr(out, "hidden_states", None)})
        return (loss, logits)


def _requested_compute_dtype(args, quantized: bool = False):
    """Resolve the requested compute dtype without silently forcing BF16.

    BF16 is used only when --bf16 is explicitly requested.  For quantized
    training, FP16 is the conservative default when neither flag is supplied.
    """
    if getattr(args, "bf16", False):
        if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported"):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "--bf16 was requested, but this CUDA device does not report BF16 support. "
                    "Use --fp16 instead."
                )
        return torch.bfloat16
    if getattr(args, "fp16", False):
        return torch.float16
    if quantized:
        return torch.float16
    return None


def _make_quant_config(args):
    use_4bit = getattr(args, "load_in_4bit", False)
    use_8bit = getattr(args, "load_in_8bit", False)
    if not (use_4bit or use_8bit):
        return None, None
    if BitsAndBytesConfig is None:
        raise ImportError(
            "BitsAndBytesConfig not available. Upgrade transformers or disable "
            "--load-in-4bit/--load-in-8bit."
        )

    # This federated runner trains one local model at a time.  Avoid
    # device_map='auto' during training because it can offload/shard modules and
    # leave LoRA adapter tensors on a different device from their activations.
    device_map = {"": 0} if torch.cuda.is_available() else None

    if use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=_requested_compute_dtype(args, quantized=True),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        return quant_cfg, device_map
    if use_8bit:
        return BitsAndBytesConfig(load_in_8bit=True), device_map
    return None, None


def _parse_target_modules(args):
    raw = getattr(args, "lora_target_modules", "q_proj,k_proj,v_proj,o_proj")
    if isinstance(raw, str):
        modules = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        modules = [str(part).strip() for part in raw if str(part).strip()]
    if not modules:
        raise ValueError("--lora-target-modules must contain at least one module name.")
    return modules


def _prepare_for_lora(backbone, args):
    """Prepare quantized/GC backbones before adding trainable LoRA adapters."""
    quantized = getattr(args, "load_in_4bit", False) or getattr(args, "load_in_8bit", False)
    use_gc = getattr(args, "grad_checkpoint", False)

    # Decoder-only training with checkpointing should not keep the KV cache.
    if hasattr(backbone, "config") and hasattr(backbone.config, "use_cache"):
        backbone.config.use_cache = False

    if quantized:
        if prepare_model_for_kbit_training is None:
            raise ImportError(
                "Quantized LoRA training requires PEFT's prepare_model_for_kbit_training. "
                "Upgrade peft (pip install -U peft) or disable quantization."
            )
        backbone = prepare_model_for_kbit_training(
            backbone,
            use_gradient_checkpointing=use_gc,
        )
    elif use_gc:
        if hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable()
        # Some decoder architectures need embedding outputs to require grads
        # when the frozen backbone is used with gradient checkpointing.
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()

    return backbone


def _apply_lora(backbone, args):
    if not getattr(args, "use_lora", False):
        return backbone
    if get_peft_model is None:
        raise ImportError("peft not installed. pip install -U peft")

    target_modules = _parse_target_modules(args)
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


def build_mistral_with_lora(model_name: str, num_labels: int, args):
    """Mistral backbone + optional quantization + optional LoRA."""
    quant_cfg, device_map = _make_quant_config(args)
    quantized = quant_cfg is not None
    model_dtype = _requested_compute_dtype(args, quantized=False)

    load_kwargs = {
        "quantization_config": quant_cfg,
        "device_map": device_map,
    }
    # Avoid passing torch_dtype=None: recent Transformers versions may warn or
    # route it differently.  For quantized models, BitsAndBytesConfig controls
    # the matmul compute dtype.
    if model_dtype is not None and not quantized:
        load_kwargs["torch_dtype"] = model_dtype

    backbone = AutoModel.from_pretrained(model_name, **load_kwargs)
    backbone = _prepare_for_lora(backbone, args)
    backbone = _apply_lora(backbone, args)

    model = DecoderClassification(backbone, num_labels=num_labels)

    # Put the classification head next to the input embeddings rather than on
    # an arbitrary first parameter.  This keeps labels/logits on the same
    # device in the federated runner and is robust for wrapped PEFT models.
    head_device = None
    try:
        emb = backbone.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            head_device = emb.weight.device
    except Exception:
        pass
    if head_device is None:
        try:
            head_device = next(p.device for p in backbone.parameters())
        except StopIteration:
            head_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.classifier = model.classifier.to(head_device)
    return model

