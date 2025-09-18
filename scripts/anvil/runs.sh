# 3b

# poisson
sbatch scripts/anvil/train_and_eval.sh configs/qwen3b/librispeech/timestamp_single/poisson+bidirectional_audio[start][bias_-6].yaml test_clean

sbatch scripts/anvil/train_and_eval.sh configs/qwen3b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][max_4].yaml test_clean 4 8
sbatch scripts/anvil/train_and_eval.sh configs/qwen3b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][max_8].yaml test_clean 8 12
sbatch scripts/anvil/train_and_eval.sh configs/qwen3b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][max_12].yaml test_clean 12 16
sbatch scripts/anvil/train_and_eval.sh configs/qwen3b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][max_16].yaml test_clean 16 20
sbatch scripts/anvil/train_and_eval.sh configs/qwen3b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][max_20].yaml test_clean 20 

# bernoulli
sbatch scripts/anvil/train_and_eval.sh configs/qwen3b/librispeech/timestamp_single/bernoulli+bidirectional_audio+class_weighting[start].yaml test_clean

sbatch scripts/anvil/train_and_eval.sh configs/qwen3b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][max_4].yaml test_clean 4 8
sbatch scripts/anvil/train_and_eval.sh configs/qwen3b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][max_8].yaml test_clean 8 12
sbatch scripts/anvil/train_and_eval.sh configs/qwen3b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][max_12].yaml test_clean 12 16
sbatch scripts/anvil/train_and_eval.sh configs/qwen3b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][max_16].yaml test_clean 16 20
sbatch scripts/anvil/train_and_eval.sh configs/qwen3b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][max_20].yaml test_clean 20 

# token
sbatch scripts/anvil/train_and_eval.sh configs/qwen3b/librispeech/timestamp_single/token+bidirectional_audio[start].yaml test_clean

sbatch scripts/anvil/train_and_eval.sh configs/qwen3b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][max_4].yaml test_clean 4 8
sbatch scripts/anvil/train_and_eval.sh configs/qwen3b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][max_8].yaml test_clean 8 12
sbatch scripts/anvil/train_and_eval.sh configs/qwen3b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][max_12].yaml test_clean 12 16
sbatch scripts/anvil/train_and_eval.sh configs/qwen3b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][max_16].yaml test_clean 16 20
sbatch scripts/anvil/train_and_eval.sh configs/qwen3b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][max_20].yaml test_clean 20 

# 7b

# poisson
sbatch scripts/anvil/train_and_eval.sh configs/qwen7b/librispeech/timestamp_single/poisson+bidirectional_audio[start][bias_-6].yaml test_clean

sbatch scripts/anvil/train_and_eval.sh configs/qwen7b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][max_4].yaml test_clean 4 8
sbatch scripts/anvil/train_and_eval.sh configs/qwen7b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][max_8].yaml test_clean 8 12
sbatch scripts/anvil/train_and_eval.sh configs/qwen7b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][max_12].yaml test_clean 12 16
sbatch scripts/anvil/train_and_eval.sh configs/qwen7b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][max_16].yaml test_clean 16 20
sbatch scripts/anvil/train_and_eval.sh configs/qwen7b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][max_20].yaml test_clean 20 

# bernoulli
sbatch scripts/anvil/train_and_eval.sh configs/qwen7b/librispeech/timestamp_single/bernoulli+bidirectional_audio+class_weighting[start].yaml test_clean

sbatch scripts/anvil/train_and_eval.sh configs/qwen7b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][max_4].yaml test_clean 4 8
sbatch scripts/anvil/train_and_eval.sh configs/qwen7b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][max_8].yaml test_clean 8 12
sbatch scripts/anvil/train_and_eval.sh configs/qwen7b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][max_12].yaml test_clean 12 16
sbatch scripts/anvil/train_and_eval.sh configs/qwen7b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][max_16].yaml test_clean 16 20
sbatch scripts/anvil/train_and_eval.sh configs/qwen7b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][max_20].yaml test_clean 20 

# token
sbatch scripts/anvil/train_and_eval.sh configs/qwen7b/librispeech/timestamp_single/token+bidirectional_audio[start].yaml test_clean

sbatch scripts/anvil/train_and_eval.sh configs/qwen7b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][max_4].yaml test_clean 4 8
sbatch scripts/anvil/train_and_eval.sh configs/qwen7b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][max_8].yaml test_clean 8 12
sbatch scripts/anvil/train_and_eval.sh configs/qwen7b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][max_12].yaml test_clean 12 16
sbatch scripts/anvil/train_and_eval.sh configs/qwen7b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][max_16].yaml test_clean 16 20
sbatch scripts/anvil/train_and_eval.sh configs/qwen7b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][max_20].yaml test_clean 20 
