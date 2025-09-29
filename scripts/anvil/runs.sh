# 3b

# librispeech

# poisson

sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_4].yaml  test  4  8
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_8].yaml  test  8  12
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_12].yaml test  12 16
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_16].yaml test  16 20
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_20].yaml test  20

# token

sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][bf16][max_4].yaml  test  4  8
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][bf16][max_8].yaml  test  8  12
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][bf16][max_12].yaml test  12 16
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][bf16][max_16].yaml test  16 20
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][bf16][max_20].yaml test  20

# bernoulli

sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_4].yaml  test  4  8
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_8].yaml  test  8  12
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_12].yaml test  12 16
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_16].yaml test  16 20
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_20].yaml test  20

# 7b

# librispeech

# poisson

sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen7b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_4].yaml  test  4  8
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen7b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_8].yaml  test  8  12
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen7b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_12].yaml test  12 16
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen7b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_16].yaml test  16 20
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen7b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_20].yaml test  20

# token

sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen7b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][bf16][max_4].yaml  test  4  8
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen7b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][bf16][max_8].yaml  test  8  12
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen7b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][bf16][max_12].yaml test  12 16
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen7b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][bf16][max_16].yaml test  16 20
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen7b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][bf16][max_20].yaml test  20

# bernoulli

sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen7b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_4].yaml  test  4  8
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen7b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_8].yaml  test  8  12
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen7b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_12].yaml test  12 16
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen7b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_16].yaml test  16 20
sbatch scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen7b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_20].yaml test  20