EXCLUDE_NODES="--exclude=h012"

# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_1e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_3e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_5e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_8e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_1e-5].yaml test

# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/token+bidirectional_audio[start][bf16][lr_1e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/token+bidirectional_audio[start][bf16][lr_3e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/token+bidirectional_audio[start][bf16][lr_5e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/token+bidirectional_audio[start][bf16][lr_8e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/token+bidirectional_audio[start][bf16][lr_1e-5].yaml test

# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_1e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_3e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_8e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_1e-5].yaml test

# sbatch $EXCLUDE_NODES scripts/anvil/eval.sh outputs/qwen3b/audioset_humans/timestamp_single/token+bidirectional_audio[start][bf16][lr_3e-6]/20251025_071933 test
# sbatch $EXCLUDE_NODES scripts/anvil/eval.sh outputs/qwen3b/audioset_humans/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_8e-6]/20251025_083825 test
# sbatch $EXCLUDE_NODES scripts/anvil/eval.sh outputs/qwen3b/audioset_humans/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_1e-5]/20251025_083825 test

# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_1e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_3e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_8e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_1e-5].yaml test

# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_1e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_1e-5].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_3e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_5e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_8e-6].yaml test

# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/token+bidirectional_audio[start][bf16][lr_1e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/token+bidirectional_audio[start][bf16][lr_3e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/token+bidirectional_audio[start][bf16][lr_5e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/token+bidirectional_audio[start][bf16][lr_8e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/token+bidirectional_audio[start][bf16][lr_1e-5].yaml test

# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen7b/librispeech/timestamp_single/poisson+bidirectional_audio[start][bias_-6][bf16][lr_1e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen7b/librispeech/timestamp_single/poisson+bidirectional_audio[start][bias_-6][bf16][lr_3e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen7b/librispeech/timestamp_single/poisson+bidirectional_audio[start][bias_-6][bf16][lr_5e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen7b/librispeech/timestamp_single/poisson+bidirectional_audio[start][bias_-6][bf16][lr_8e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen7b/librispeech/timestamp_single/poisson+bidirectional_audio[start][bias_-6][bf16][lr_1e-5].yaml test

# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen7b/librispeech/timestamp_single/token+bidirectional_audio[start][bf16][lr_1e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen7b/librispeech/timestamp_single/token+bidirectional_audio[start][bf16][lr_3e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen7b/librispeech/timestamp_single/token+bidirectional_audio[start][bf16][lr_5e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen7b/librispeech/timestamp_single/token+bidirectional_audio[start][bf16][lr_8e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen7b/librispeech/timestamp_single/token+bidirectional_audio[start][bf16][lr_1e-5].yaml test

# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6][og_prompt].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_5e-6][og_prompt].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/token+bidirectional_audio[start][bf16][lr_5e-6][og_prompt].yaml test

# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_1e-6][og_prompt][epoch_3].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_1e-6][og_prompt][epoch_3].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/token+bidirectional_audio[start][bf16][lr_1e-6][og_prompt][epoch_3].yaml test

# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/token+bidirectional_audio[start][bf16][lr_1e-6][og_prompt][epoch_3][no_schedule].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_1e-6][og_prompt][epoch_3][no_schedule].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_1e-6][og_prompt][epoch_3][no_schedule].yaml test

# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/bernoulli+class_weighting+bidirectional_audio[start][bf16][lr_1e-6][og_prompt][epoch_3][no_schedule].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/bernoulli+class_weighting+bidirectional_audio[start][bf16][lr_5e-6][og_prompt].yaml test

# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/bernoulli+class_weighting+bidirectional_audio[start][bf16][upscale_4][lr_1e-6][og_prompt][epoch_3][no_schedule].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/bernoulli+class_weighting+bidirectional_audio[start][bf16][upscale_4][lr_5e-6][og_prompt].yaml test

# audioset_humans
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/token+bidirectional_audio[start][bf16][lr_5e-6].yaml test
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6].yaml test
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_5e-6].yaml test
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/bernoulli+class_weighting+bidirectional_audio[start][bf16][lr_5e-6].yaml test
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/bernoulli+class_weighting+bidirectional_audio[start][bf16][upscale_4][lr_5e-6].yaml test

# joint_token_poisson libricount
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/token+poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6][weight_0.05].yaml test
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/token+poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6][weight_0.1].yaml test
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/token+poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6][weight_0.2].yaml test
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/token+poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6][weight_0.5].yaml test
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/token+poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6][weight_1.0].yaml test
