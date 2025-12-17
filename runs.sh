EXCLUDE_NODES="--exclude=h012,h018"

# ----------- FINE TUNES -----------

# Qwen 2.5 Omni 3B

# LibriSpeech Clean

# 1e-6 lr
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/new_inference/token[start][bf16][lr_1e-6][epoch_3][no_schedule].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/new_inference/poisson[start][bias_-6][bf16][upscale_4][lr_1e-6][epoch_3][no_schedule].yaml

# LibriCount

# 1e-6 lr
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/libricount/timestamp_any/new_inference/token[start][bf16][lr_1e-6][no_schedule][epoch_3].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/libricount/timestamp_any/new_inference/poisson[start][bias_-6.9][bf16][upscale_4][lr_1e-6][no_schedule][epoch_3].yaml

# Audioset Humans

# 1e-6 lr
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/audioset_humans_resampled/timestamp_any/token[start][bf16][lr_1e-6][epoch_3][no_schedule].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/audioset_humans_resampled/timestamp_any/poisson[start][bias_-6][bf16][upscale_4][lr_1e-6][epoch_3][no_schedule].yaml

# Qwen 2.5 Omni 7B

# LibriSpeech Clean

# 1e-6 lr
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/new_inference/token[start][bf16][lr_1e-6][epoch_3][no_schedule].yaml 
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/new_inference/poisson[start][bias_-6][bf16][upscale_4][lr_1e-6][epoch_3][no_schedule].yaml

# LibriCount

# 1e-6 lr
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/new_inference/token[start][bf16][lr_1e-6][no_schedule][epoch_6].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/new_inference/poisson[start][bias_-6.9][bf16][upscale_4][lr_1e-6][no_schedule][epoch_6].yaml

# Audioset Humans

# 1e-6 lr
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/audioset_humans_resampled/timestamp_any/token[start][bf16][lr_1e-6][epoch_6][no_schedule].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/audioset_humans_resampled/timestamp_any/poisson[start][bias_-6][bf16][upscale_4][lr_1e-6][epoch_6][no_schedule].yaml

# ----------- ABLATIONS -----------

sbatch scripts/hyak/accelerate_ga_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_4].yaml 4 8