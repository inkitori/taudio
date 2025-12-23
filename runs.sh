EXCLUDE_NODES="--exclude=h012,h018"

# ----------- FINE TUNES -----------

# Qwen 2.5 Omni 3B

# LibriSpeech Clean

# 1e-6 lr
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/new_inference/token[start][bf16][lr_1e-6][epoch_3][no_schedule].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/new_inference/poisson[start][bias_-6][bf16][upscale_4][lr_1e-6][epoch_3][no_schedule].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_3][no_schedule][lr_1e-6].yaml

# LibriCount

# 1e-6 lr
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/libricount/timestamp_any/new_inference/token[start][bf16][lr_1e-6][no_schedule][epoch_3].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/libricount/timestamp_any/new_inference/poisson[start][bias_-6.9][bf16][upscale_4][lr_1e-6][no_schedule][epoch_3].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/libricount/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_3][no_schedule][lr_1e-6].yaml

# Audioset Humans

# 1e-6 lr
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/audioset_humans_resampled/timestamp_any/token[start][bf16][lr_1e-6][epoch_3][no_schedule].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/audioset_humans_resampled/timestamp_any/poisson[start][bias_-6][bf16][upscale_4][lr_1e-6][epoch_3][no_schedule].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/audioset_humans_resampled/timestamp_any/binary[class_weighting][start][bf16][upscale_4][lr_1e-6][epoch_3].yaml

# Qwen 2.5 Omni 7B

# LibriSpeech Clean

# 1e-6 lr
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/new_inference/token[start][bf16][lr_1e-6][epoch_3][no_schedule].yaml 
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/new_inference/poisson[start][bias_-6][bf16][upscale_4][lr_1e-6][epoch_3][no_schedule].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_3][no_schedule][lr_1e-6].yaml

# LibriCount

# 1e-6 lr
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/new_inference/token[start][bf16][lr_1e-6][no_schedule][epoch_6].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/new_inference/poisson[start][bias_-6.9][bf16][upscale_4][lr_1e-6][no_schedule][epoch_6].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_6][no_schedule][lr_1e-6].yaml

# Audioset Humans

# 1e-6 lr
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/audioset_humans_resampled/timestamp_any/token[start][bf16][lr_1e-6][epoch_6][no_schedule].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/audioset_humans_resampled/timestamp_any/poisson[start][bias_-6][bf16][upscale_4][lr_1e-6][epoch_6][no_schedule].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/audioset_humans_resampled/timestamp_any/binary[class_weighting][start][bf16][upscale_4][lr_1e-6][epoch_6].yaml

# ----------- ABLATIONS -----------

# Tokens
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_4].yaml 4 8
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_8].yaml 8 12
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_12].yaml 12 16
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_16].yaml 16 20
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_20].yaml 20

# Poisson
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_4].yaml 4 8
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_8].yaml 8 12
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_12].yaml 12 16
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_16].yaml 16 20
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_20].yaml 20

# Binary
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_4].yaml 4 8
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_8].yaml 8 12
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_12].yaml 12 16
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_16].yaml 16 20
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_20].yaml 20

# ----------- JOINT TRAINING -----------
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/new_inference/token+poisson[start][bias_-6][bf16][upscale_4][lr_1e-6][epoch_3][no_schedule].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/libricount/timestamp_any/new_inference/token+poisson[start][bias_-6.9][bf16][upscale_4][lr_1e-6][no_schedule][epoch_3].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/audioset_humans_resampled/timestamp_any/token+poisson[start][bias_-6][bf16][upscale_4][lr_1e-6][epoch_3][no_schedule].yaml