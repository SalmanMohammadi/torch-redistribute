# torch-redistribute

This repo demonstrates a technique for redistributing the weights of a model sharded using FSDP to a tensor parallel configuration,
without duplicating the model weights. 

This is achieved by re-sharding the FSDP model weights, and using another `generation_model` which is able to *share* the re-sharded weights. 
This second model also has the necessary pre-and-post-forward hooks for the relevant tensor parallel styles. This means we don't need to mess around with the original model's pre/post forward/backward hooks.

## Usage

```python
torchrun --nnodes=1 --nproc-per-node=2 --standalone redistribute_llama.py 
```