#!/usr/bin/env python3
"""Fix mma.h PTX inline assembly for MUSA build."""
import re

filepath = "/workspace/tiny-cuda-nn/include_musa/tiny-cuda-nn/mma.h"

with open(filepath, "r") as f:
    content = f.read()

# Replace the entire mma_sync_accum method with a disabled version
# The PTX mma.sync and wmma instructions are NVIDIA-specific
content = content.replace(
    'asm(\n\t\t\t"mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "',
    '// DISABLED FOR MUSA: PTX mma.sync not available\n\t\t// asm(\n\t\t//\t"mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "'
)

content = content.replace(
    'asm(\n\t\t\t"wmma.mma.sync.aligned.row.col.m16n16k16.f16.f16 "',
    '// DISABLED FOR MUSA: PTX wmma not available\n\t\t// asm(\n\t\t//\t"wmma.mma.sync.aligned.row.col.m16n16k16.f16.f16 "'
)

with open(filepath, "w") as f:
    f.write(content)

print("mma.h PTX instructions disabled for MUSA")
