"""
rLLM Tools - Utilities for data processing and model management.

Tools available:
- convert_to_qwen: Convert trajectories to Qwen format for SFT
- view_parquet: View parquet file contents
- convert_fsdp_to_hf: Convert FSDP checkpoints to HuggingFace format
- chat_with_checkpoint: Interactive chat with a trained checkpoint
- generate_sft_data: Generate SFT training data from trajectories

Usage:
    python -m src.tools.convert_to_qwen --input data.parquet --output data_qwen.parquet
    python -m src.tools.view_parquet --parquet_file data.parquet
    python -m src.tools.convert_fsdp_to_hf --checkpoint path/to/ckpt --output path/to/hf
    python -m src.tools.chat_with_checkpoint --model path/to/model
"""
