"""Report the torch / CUDA / diffusers versions this env resolved to."""

import torch
import diffusers
import transformers

print(f"torch        {torch.__version__}")
print(f"diffusers    {diffusers.__version__}")
print(f"transformers {transformers.__version__}")
print(f"cuda avail   {torch.cuda.is_available()}")
print(f"cuda build   {torch.version.cuda}")
print(f"bf16 support {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else 'n/a'}")

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  gpu {i}: {props.name}  {props.total_memory / 1e9:.0f}GB  sm_{props.major}{props.minor}")

from diffusers import QwenImageEditPlusPipeline  # noqa: E402

print(f"QwenImageEditPlusPipeline OK: {QwenImageEditPlusPipeline}")
