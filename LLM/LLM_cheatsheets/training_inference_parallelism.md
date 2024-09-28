|     |  Point of split   |   pros  | cons |  available package   |
| --- | --- | --- | --- | --- |
| Data Parallelization   | data batches   | no model change  | OOM when model too large | [torch](https://pytorch.org/tutorials/beginner/ddp_series_theory.html)  |
| Pipeline parallelization | layers | keep the head integrity | - idle devices waiting <br> - memory imbalance| - Torch <br> - [Colossal-AI](https://colossalai.org/docs/features/pipeline_parallel/) <br> - [Nvidia-Megatron](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/pipeline_parallel) <br> - [DeepSpeed](https://www.deepspeed.ai/tutorials/pipeline/) |
| Tensor parallelization   | attention heads/hidden dimension | memory efficient | can't apply to LayerNorm and Dropouts | - [Torch](https://pytorch.org/tutorials/intermediate/TP_tutorial.html) <br> - [Colossal-AI](https://colossalai.org/docs/features/1D_tensor_parallel) <br> - [Nvidia-Megatron](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/parallelisms.html) |
| Sequence parallelization | sequence dimension | solves tensor parallelization issues | only applied to sequence-independent operations| |
| Zero | optimizer state | reduces optimizer memory | optimizer got altered| [DeepSpeed](https://huggingface.co/docs/accelerate/v0.11.0/en/deepspeed)|
| Zero-offload | optimizer state between GPU/CPU/Disk/NVMe | compensate GPU limits | communication overhead  | [DeepSpeed](https://huggingface.co/docs/accelerate/v0.11.0/en/deepspeed) |
| Ring Attention | | | | 

Reference:
https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
https://pytorch.org/docs/stable/distributed.pipelining.html#pipeline-parallelism
https://huggingface.co/docs/transformers/v4.15.0/en/parallelism#naive-model-parallel-vertical-and-pipeline-parallel
https://huggingface.co/blog/bloom-megatron-deepspeed#pipeline-parallelism
https://developer.habana.ai/blog/training-llama-and-bloom-13-billion-parameter-llms-with-3d-parallelism-on-habana-gaudi2/
