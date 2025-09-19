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


sbatch scripts/anvil/eval.sh outputs/qwen3b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][max_4]/20250918_082554 test_clean 4 8
sbatch scripts/anvil/eval.sh outputs/qwen3b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][max_8]/20250918_085013 test_clean 8 12
sbatch scripts/anvil/eval.sh outputs/qwen3b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][max_12]/20250918_085424 test_clean 12 16
sbatch scripts/anvil/eval.sh outputs/qwen3b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][max_16]/20250918_085830 test_clean 16 20
sbatch scripts/anvil/eval.sh outputs/qwen3b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][max_20]/20250918_102345 test_clean 20

sbatch scripts/anvil/eval.sh outputs/qwen3b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][max_4]/20250918_041908 test_clean 4 8
sbatch scripts/anvil/eval.sh outputs/qwen3b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][max_8]/20250918_041910 test_clean 8 12
sbatch scripts/anvil/eval.sh outputs/qwen3b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][max_12]/20250918_054720 test_clean 12 16
sbatch scripts/anvil/eval.sh outputs/qwen3b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][max_16]/20250918_060337 test_clean 16 20
sbatch scripts/anvil/eval.sh outputs/qwen3b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][max_20]/20250918_060943 test_clean 20

sbatch scripts/anvil/eval.sh outputs/qwen3b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][max_4]/20250918_063405 test_clean 4 8
sbatch scripts/anvil/eval.sh outputs/qwen3b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][max_8]/20250918_063809 test_clean 8 12
sbatch scripts/anvil/eval.sh outputs/qwen3b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][max_12]/20250918_064011 test_clean 12 16
sbatch scripts/anvil/eval.sh outputs/qwen3b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][max_16]/20250918_080538 test_clean 16 20
sbatch scripts/anvil/eval.sh outputs/qwen3b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][max_20]/20250918_081947 test_clean 20

sbatch scripts/anvil/eval.sh outputs/qwen7b/librispeech/timestamp_single/bernoulli+bidirectional_audio+class_weighting[start]/20250918_104003 test_clean
sbatch scripts/anvil/eval.sh outputs/qwen7b/librispeech/timestamp_single/poisson+bidirectional_audio[start][bias_-6]/20250918_103602 test_clean
sbatch scripts/anvil/eval.sh outputs/qwen7b/librispeech/timestamp_single/token+bidirectional_audio[start]/20250918_110634 test_clean

sbatch scripts/anvil/eval.sh outputs/qwen7b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][max_4]/20250918_133046 test_clean 4 8
sbatch scripts/anvil/eval.sh outputs/qwen7b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][max_8]/20250918_133258 test_clean 8 12
sbatch scripts/anvil/eval.sh outputs/qwen7b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][max_12]/20250918_133655 test_clean 12 16
sbatch scripts/anvil/eval.sh outputs/qwen7b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][max_16]/20250918_150820 test_clean 16 20
sbatch scripts/anvil/eval.sh outputs/qwen7b/librispeech/timestamp_single/ablation/bernoulli+bidirectional_audio+class_weighting[start][max_20]/20250918_152434 test_clean 20

sbatch scripts/anvil/eval.sh outputs/qwen7b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][max_4]/20250918_110836 test_clean 4 8
sbatch scripts/anvil/eval.sh outputs/qwen7b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][max_8]/20250918_111240 test_clean 8 12
sbatch scripts/anvil/eval.sh outputs/qwen7b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][max_12]/20250918_124203 test_clean 12 16
sbatch scripts/anvil/eval.sh outputs/qwen7b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][max_16]/20250918_130017 test_clean 16 20
sbatch scripts/anvil/eval.sh outputs/qwen7b/librispeech/timestamp_single/ablation/poisson+bidirectional_audio[start][bias_-6][max_20]/20250918_130420 test_clean 20