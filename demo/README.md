# Adaptive MoE demo

This package keeps demo code outside the research notebook. It does not claim a live model result until the real-expert quantization compatibility gate is passed.

## Current environment status

The checkout currently has no `model/agent.zip`; its policy archives are `model/best_model.zip` and `model/rl_agent_final (1).zip`. The current host has no CUDA GPU and no installed PyTorch, Transformers, bitsandbytes, or Gradio. The live backend therefore fails closed and the replay UI reports only values present in uploaded records. Do not label hypothetical or synthetic timing/memory values as GPU measurements.

The existing notebook uses `allenai/OLMoE-1B-7B-0924-Instruct`, raw 64 routing logits followed by `[battery/100, 4/8 RAM, 0.10 drain, 0 thermal]`, and PPO tiers FP16/FP8/FP4. The supplied demo target/model plan specifies a different base model and NF4/INT8/FP16. Those are not interchangeable without revalidating the checkpoint dimensions, feature construction, action mapping, and trained policy. The PPO checkpoint must be explicitly selected and hashed for a live run.

## Launch

Replay mode: `./demo/launch-replay.sh` (or install the replay requirements and run `python -m demo.app`). A replay requires an exported `RunRecord` JSON; this repository currently contains no validated GPU recording.

GPU host: `./demo/launch-gpu.sh`. It currently refuses live inference by design. Before enabling that path, complete and retain evidence for the plan's one-expert FP16/INT8/NF4 gate, then full-model routing-count agreement, peak memory and budget enforcement, cancellation, repeated-request stability, and OOM recovery. The OLMoE 4.57.1 module API and bitsandbytes layer behavior must be confirmed on the actual GPU and frozen dependencies.

For a remote GPU, bind to localhost and use SSH port forwarding (`ssh -L 7860:127.0.0.1:7860 user@gpu-host`). Do not expose an unauthenticated Gradio service publicly.

## Data contracts and exports

`RunRequest`, `RunEvent`, and `RunRecord` are in `contracts.py`. `replay.py` loads records and exports session JSON/CSV. Replay is read-only and displays the stored status, output, event stream, and measurements without launching inference.

The plan's frozen correctness suite and full live comparisons are intentionally not represented as completed: they require the backend and GPU validation first. This prevents the UI from implying that the proposed policy has been measured or outperforms a baseline.
