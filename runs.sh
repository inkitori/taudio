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
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/token+bidirectional_audio[start][bf16][lr_5e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_5e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/bernoulli+class_weighting+bidirectional_audio[start][bf16][lr_5e-6].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/bernoulli+class_weighting+bidirectional_audio[start][bf16][upscale_4][lr_5e-6].yaml test

# joint_token_poisson libricount
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/token+poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6][weight_0.05].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/token+poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6][weight_0.1].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/token+poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6][weight_0.2].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/token+poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6][weight_0.5].yaml test
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/libricount/timestamp_single/token+poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6][weight_1.0].yaml test

# joint_token_poisson audioset_humans
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16.sh configs/qwen3b/audioset_humans/timestamp_single/token+poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6][weight_1.0].yaml test

# unified audioset humans timestamp single
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/audioset_humans/timestamp_single/token+bidirectional_audio[start][bf16][lr_5e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/audioset_humans/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/audioset_humans/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_5e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/audioset_humans/timestamp_single/bernoulli+class_weighting+bidirectional_audio[start][bf16][lr_5e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/audioset_humans/timestamp_single/bernoulli+class_weighting+bidirectional_audio[start][bf16][upscale_4][lr_5e-6].yaml

# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/audioset_humans/timestamp_single_any/token+bidirectional_audio[start][bf16][lr_5e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/audioset_humans/timestamp_single_any/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/audioset_humans/timestamp_single_any/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_5e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/audioset_humans/timestamp_single_any/bernoulli+class_weighting+bidirectional_audio[start][bf16][lr_5e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/audioset_humans/timestamp_single_any/bernoulli+class_weighting+bidirectional_audio[start][bf16][upscale_4][lr_5e-6].yaml

# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/groove_midi/timestamp_single_any/token+bidirectional_audio[start][bf16][lr_5e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/groove_midi/timestamp_single_any/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/groove_midi/timestamp_single_any/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_5e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/groove_midi/timestamp_single_any/bernoulli+class_weighting+bidirectional_audio[start][bf16][lr_5e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/groove_midi/timestamp_single_any/bernoulli+class_weighting+bidirectional_audio[start][bf16][upscale_4][lr_5e-6].yaml


# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_4].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_8].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_12].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_16].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_20].yaml

# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_4].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_8].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_12].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_16].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_20].yaml

# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_4].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_8].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_12].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_16].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_20].yaml


# qwen 3b librispeech (no ablations) any
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/token+bidirectional_audio[start][bf16].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/token+poisson+bidirectional_audio[start][bias_-6][bf16].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/poisson+bidirectional_audio[start][bias_-6][bf16].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/poisson+bidirectional_audio[start][bias_-6][bf16][upscale_4].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/bernoulli+bidirectional_audio+class_weighting[start][bf16].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/bernoulli+bidirectional_audio+class_weighting[start][bf16][upscale_4].yaml

# qwen 7b librispeech (no ablations) any
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/token+bidirectional_audio[start][bf16].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/token+poisson+bidirectional_audio[start][bias_-6][bf16].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/poisson+bidirectional_audio[start][bias_-6][bf16].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/poisson+bidirectional_audio[start][bias_-6][bf16][upscale_4].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/bernoulli+bidirectional_audio+class_weighting[start][bf16].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/bernoulli+bidirectional_audio+class_weighting[start][bf16][upscale_4].yaml

# qwen 3b librispeech ablations any
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_4].yaml 4 8
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_8].yaml 8 12
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_12].yaml 12 16
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_16].yaml 16 20
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_20].yaml 20

# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_4].yaml 4 8
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_8].yaml 8 12
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_12].yaml 12 16
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_16].yaml 16 20
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_20].yaml 20

# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_4].yaml 4 8
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_8].yaml 8 12
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_12].yaml 12 16
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_16].yaml 16 20
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_20].yaml 20

# qwen 7b librispeech ablations any
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_4].yaml 4 8
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_8].yaml 8 12
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_12].yaml 12 16
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_16].yaml 16 20
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_20].yaml 20

# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_4].yaml 4 8
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_8].yaml 8 12
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_12].yaml 12 16
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_16].yaml 16 20
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_20].yaml 20

# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_4].yaml 4 8
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_8].yaml 8 12
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_12].yaml 12 16
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_16].yaml 16 20
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_20].yaml 20

# qwen 7b librispeech ablations single
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][bf16][max_4].yaml 4 8
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][bf16][max_8].yaml 8 12
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][bf16][max_12].yaml 12 16
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][bf16][max_16].yaml 16 20
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_single/ablation/token+bidirectional_audio[start][bf16][max_20].yaml 20

# qwen 7b librispeech ablations any (tuning poisson parameters)
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/tuning/poisson+bidirectional_audio[start][bias_-6][bf16][max_4][lr_1e-6].yaml 4 8
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/tuning/poisson+bidirectional_audio[start][bias_-6][bf16][max_4][lr_2e-6].yaml 4 8
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/tuning/poisson+bidirectional_audio[start][bias_-6][bf16][max_4][lr_3e-6].yaml 4 8
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/tuning/poisson+bidirectional_audio[start][bias_-6][bf16][max_4][lr_7e-6].yaml 4 8
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/tuning/poisson+bidirectional_audio[start][bias_-6][bf16][max_4][lr_8e-6].yaml 4 8
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/tuning/poisson+bidirectional_audio[start][bias_-6][bf16][max_4][lr_1e-5].yaml 4 8

# # qwen 7b librispeech any lr 2e-6
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/token+bidirectional_audio[start][bf16][lr_2e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/poisson+bidirectional_audio[start][bias_-6][bf16][lr_2e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/poisson+bidirectional_audio[start][bias_-6][bf16][upscale_4][lr_2e-6].yaml
# # poisson ablations
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_4][lr_2e-6].yaml 4 8
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_8][lr_2e-6].yaml 8 12
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_12][lr_2e-6].yaml 12 16
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_16][lr_2e-6].yaml 16 20
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_20][lr_2e-6].yaml 20

# # qwen 7b librispeech any lr 3e-6
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/token+bidirectional_audio[start][bf16][lr_3e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/poisson+bidirectional_audio[start][bias_-6][bf16][lr_3e-6].yaml

# # qwen 7b librispeech any lr 1e-6
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/token+bidirectional_audio[start][bf16][lr_1e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/poisson+bidirectional_audio[start][bias_-6][bf16][lr_1e-6].yaml

# # qwen 7b librispeech any lr 7e-6
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/token+bidirectional_audio[start][bf16][lr_7e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/poisson+bidirectional_audio[start][bias_-6][bf16][lr_7e-6].yaml

# # qwen 7b librispeech any poisson lr 1e-6 ablations
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_4][lr_1e-6].yaml 4 8
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_8][lr_1e-6].yaml 8 12
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_12][lr_1e-6].yaml 12 16
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_16][lr_1e-6].yaml 16 20
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_20][lr_1e-6].yaml 20

# # token ablations
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_4][lr_1e-6].yaml 4 8
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_8][lr_1e-6].yaml 8 12
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_12][lr_1e-6].yaml 12 16
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_16][lr_1e-6].yaml 16 20
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_20][lr_1e-6].yaml 20

# # qwen 7b libricount any 
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/token+bidirectional_audio[start][bf16][lr_2e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_2e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_2e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/bernoulli+class_weighting+bidirectional_audio[start][bf16][lr_2e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/bernoulli+class_weighting+bidirectional_audio[start][bf16][upscale_4][lr_2e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/token+poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_2e-6][weight_1.0].yaml

# # qwen 7b libricount any lr 5e-6
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/token+bidirectional_audio[start][bf16][lr_5e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_5e-6].yaml

# # qwen 7b audioset_humans any
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/audioset_humans/timestamp_any/token+bidirectional_audio[start][bf16][lr_2e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/audioset_humans/timestamp_any/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_2e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/audioset_humans/timestamp_any/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_2e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/audioset_humans/timestamp_any/bernoulli+class_weighting+bidirectional_audio[start][bf16][lr_2e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/audioset_humans/timestamp_any/bernoulli+class_weighting+bidirectional_audio[start][bf16][upscale_4][lr_2e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/audioset_humans/timestamp_any/token+poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_2e-6][weight_1.0].yaml

# # qwen 7b audioset_humans any lr 5e-6
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/audioset_humans/timestamp_any/token+bidirectional_audio[start][bf16][lr_5e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/audioset_humans/timestamp_any/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/audioset_humans/timestamp_any/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_5e-6].yaml

# # qwen 7b libricount any tuning
# # poisson
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/tuning/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_1e-6][no_schedule].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/tuning/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_2e-6][no_schedule].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/tuning/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_3e-6][no_schedule].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/tuning/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_4e-6][no_schedule].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/tuning/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6][no_schedule].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/tuning/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_6e-6][no_schedule].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/tuning/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_7e-6][no_schedule].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/tuning/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_8e-6][no_schedule].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/tuning/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_9e-6][no_schedule].yaml
# # fp16
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_fp16_run.sh configs/qwen7b/libricount/timestamp_any/tuning/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_2e-6][no_schedule][fp16].yaml

# # poisson frame upscaling
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/tuning/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_1e-6][no_schedule].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/tuning/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_2e-6][no_schedule].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/tuning/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_5e-6][no_schedule].yaml

# # token
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/tuning/token+bidirectional_audio[start][bf16][lr_1e-6][no_schedule].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/tuning/token+bidirectional_audio[start][bf16][lr_2e-6][no_schedule].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/tuning/token+bidirectional_audio[start][bf16][lr_5e-6][no_schedule].yaml

# # qwen 3b libricount any
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/libricount/timestamp_single/token+bidirectional_audio[start][bf16][lr_5e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/libricount/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/libricount/timestamp_single/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_5e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/libricount/timestamp_single/bernoulli+class_weighting+bidirectional_audio[start][bf16][lr_5e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/libricount/timestamp_single/bernoulli+class_weighting+bidirectional_audio[start][bf16][upscale_4][lr_5e-6].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/libricount/timestamp_single/token+poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6][weight_1.0].yaml

# # qwen 7b libricount any tuning (epochs)
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/tuning/token+bidirectional_audio[start][bf16][lr_5e-6][epoch_5].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/tuning/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6][epoch_5].yaml
# sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen7b/libricount/timestamp_any/tuning/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6][upscale_4][epoch_5].yaml

# qwen 3b tuning 5 epochs
# librispeech
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/tuning/token+bidirectional_audio[start][bf16][epoch_5].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/tuning/poisson+bidirectional_audio[start][bias_-6][bf16][epoch_5].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/tuning/poisson+bidirectional_audio[start][bias_-6][bf16][upscale_4][epoch_5].yaml

# libricount
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/libricount/timestamp_single/tuning/token+bidirectional_audio[start][bf16][lr_5e-6][epoch_5].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/libricount/timestamp_single/tuning/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6][epoch_5].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/libricount/timestamp_single/tuning/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_5e-6][epoch_5].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/audioset_humans/timestamp_any/tuning/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6][epoch_5][pad_0.5].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/audioset_humans/timestamp_any/tuning/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_5e-6][epoch_5][pad_0.5].yaml

# audioset_humans
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/audioset_humans/timestamp_any/tuning/token+bidirectional_audio[start][bf16][lr_5e-6][epoch_5].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/audioset_humans/timestamp_any/tuning/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6][epoch_5].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/audioset_humans/timestamp_any/tuning/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_5e-6][epoch_5].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/libricount/timestamp_single/tuning/poisson+bidirectional_audio[start][bias_-6.9][bf16][lr_5e-6][epoch_5][pad_0.5].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/libricount/timestamp_single/tuning/poisson+bidirectional_audio[start][bias_-6.9][bf16][upscale_4][lr_5e-6][epoch_5][pad_0.5].yaml