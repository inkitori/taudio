sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_1e-6].yaml test
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_3e-6].yaml test
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_5e-6].yaml test
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_8e-6].yaml test
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_1e-5].yaml test

sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/token+bidirectional_audio[start][bf16][lr_1e-6].yaml test
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/token+bidirectional_audio[start][bf16][lr_3e-6].yaml test
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/token+bidirectional_audio[start][bf16][lr_5e-6].yaml test
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/token+bidirectional_audio[start][bf16][lr_8e-6].yaml test
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/token+bidirectional_audio[start][bf16][lr_1e-5].yaml test