# Verl tutorial

reference: 
- [official doc](https://verl.readthedocs.io/en/latest/start/multinode.html#option-3-launch-via-slurm)
- [Verlフレームワークを用いたLLMの強化学習(PPO,GRPO,DAPO)](https://zenn.dev/puwa/articles/verl_document_report)

## verl with singularity
the containers in [verl docker hub](https://hub.docker.com/r/verlai/verl/tags) already have the env including ray installed. we just need to transfer it to singularity format and use. if sudo is not available for yout user, we should build a sandbox and use it

steps:
- singularity build sif_file_path docker://verlai/verl:vllm012.latest
    - (or) singularity build --sandbox sandbox_path docker://verlai/verl:vllm012.latest
- (only for sandbox) singularity run --writable --containall --no-home sandbox_path
- setup the env params and bindings according to [multinode example](https://github.com/volcengine/verl/blob/main/examples/slurm/ray_on_slurm.slurm)
- debug to fix all the things in your script
- train the model

