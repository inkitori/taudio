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

# lambda=1
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/new_inference/token+poisson[start][bias_-6][bf16][upscale_4][lr_1e-6][epoch_3][no_schedule].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/libricount/timestamp_any/new_inference/token+poisson[start][bias_-6.9][bf16][upscale_4][lr_1e-6][no_schedule][epoch_3].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/audioset_humans_resampled/timestamp_any/token+poisson[start][bias_-6][bf16][upscale_4][lr_1e-6][epoch_3][no_schedule].yaml

# lambda=0.05
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_any/new_inference/token+poisson[start][bias_-6][bf16][upscale_4][lr_1e-6][epoch_3][no_schedule][lambda=5e-2].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/libricount/timestamp_any/new_inference/token+poisson[start][bias_-6.9][bf16][upscale_4][lr_1e-6][no_schedule][epoch_3][lambda=5e-2].yaml
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_4_gpu_bf16_run.sh configs/qwen3b/audioset_humans_resampled/timestamp_any/token+poisson[start][bias_-6][bf16][upscale_4][lr_1e-6][epoch_3][no_schedule][lambda=5e-2].yaml

# ----------- MULTI TIMESTAMPS -----------
# 3B
sbatch $EXCLUDE_NODES scripts/anvil/ga_accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_all/token[start][bf16][lr_1e-6][epoch_3][no_schedule].yaml
sbatch $EXCLUDE_NODES scripts/anvil/ga_accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_all/poisson[start][bias_-6][bf16][upscale_4][lr_1e-6][epoch_3][no_schedule].yaml

# higher frame resolutions for poisson
sbatch $EXCLUDE_NODES scripts/anvil/ga_accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_all/poisson[start][bias_-6][bf16][upscale_10][lr_1e-6][epoch_3][no_schedule].yaml

# no frame resolutions for poisson
sbatch $EXCLUDE_NODES scripts/anvil/ga_accelerate_4_gpu_bf16_run.sh configs/qwen3b/librispeech/timestamp_all/poisson[start][bias_-6][bf16][upscale_1][lr_1e-6][epoch_1][no_schedule].yaml 

# 7B
sbatch $EXCLUDE_NODES scripts/anvil/ga_accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_all/token[start][bf16][lr_1e-6][epoch_3][no_schedule].yaml
sbatch $EXCLUDE_NODES scripts/anvil/ga_accelerate_4_gpu_bf16_run.sh configs/qwen7b/librispeech/timestamp_all/poisson[start][bias_-6][bf16][upscale_4][lr_1e-6][epoch_3][no_schedule].yaml

# ----------- BERNOULLI DEV SWEEP -----------

# 3B
# Librispeech
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_3][no_schedule][lr_1e-6].yaml outputs/qwen3b/librispeech/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_3][no_schedule][lr_1e-6]/20251221_235944/checkpoint_epoch3 dev
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_3][no_schedule][lr_1e-6].yaml outputs/qwen3b/librispeech/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_3][no_schedule][lr_1e-6]/20251221_235944/checkpoint_epoch2 dev
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_3][no_schedule][lr_1e-6].yaml outputs/qwen3b/librispeech/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_3][no_schedule][lr_1e-6]/20251221_235944/checkpoint_epoch1 dev

# after picking the best, epoch 3 does the best on dev/smooth_20ms_boxcar_broken/aux_correct_20ms

# Ablations
# < 4
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_4].yaml outputs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_4]/20251223_061132/checkpoint_epoch3 dev 
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_4].yaml outputs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_4]/20251223_061132/checkpoint_epoch2 dev
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_4].yaml outputs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_4]/20251223_061132/checkpoint_epoch1 dev

# < 8
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_8].yaml outputs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_8]/20251223_061142/checkpoint_epoch3 dev
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_8].yaml outputs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_8]/20251223_061142/checkpoint_epoch2 dev
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_8].yaml outputs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_8]/20251223_061142/checkpoint_epoch1 dev

# < 12
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_12].yaml outputs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_12]/20251223_061142/checkpoint_epoch3 dev
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_12].yaml outputs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_12]/20251223_061142/checkpoint_epoch2 dev
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_12].yaml outputs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_12]/20251223_061142/checkpoint_epoch1 dev

# < 16
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_16].yaml outputs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_16]/20251223_084333/checkpoint_epoch3 dev
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_16].yaml outputs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_16]/20251223_084333/checkpoint_epoch2 dev
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_16].yaml outputs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_16]/20251223_084333/checkpoint_epoch1 dev

# < 20
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_20].yaml outputs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_20]/20251223_084347/checkpoint_epoch3 dev
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_20].yaml outputs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_20]/20251223_084347/checkpoint_epoch2 dev
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_20].yaml outputs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_20]/20251223_084347/checkpoint_epoch1 dev

# Libricount
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/libricount/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_3][no_schedule][lr_1e-6].yaml outputs/qwen3b/libricount/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_3][no_schedule][lr_1e-6]/20251222_014534/checkpoint_epoch3 dev
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/libricount/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_3][no_schedule][lr_1e-6].yaml outputs/qwen3b/libricount/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_3][no_schedule][lr_1e-6]/20251222_014534/checkpoint_epoch2 dev
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/libricount/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_3][no_schedule][lr_1e-6].yaml outputs/qwen3b/libricount/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_3][no_schedule][lr_1e-6]/20251222_014534/checkpoint_epoch1 dev

# Audioset
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/audioset_humans_resampled/timestamp_any/binary[class_weighting][start][bf16][upscale_4][lr_1e-6][epoch_3].yaml outputs/qwen3b/audioset_humans_resampled/timestamp_any/binary[class_weighting][start][bf16][upscale_4][lr_1e-6][epoch_3]/20251222_042343/checkpoint_epoch3 dev
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/audioset_humans_resampled/timestamp_any/binary[class_weighting][start][bf16][upscale_4][lr_1e-6][epoch_3].yaml outputs/qwen3b/audioset_humans_resampled/timestamp_any/binary[class_weighting][start][bf16][upscale_4][lr_1e-6][epoch_3]/20251222_042343/checkpoint_epoch2 dev
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/audioset_humans_resampled/timestamp_any/binary[class_weighting][start][bf16][upscale_4][lr_1e-6][epoch_3].yaml outputs/qwen3b/audioset_humans_resampled/timestamp_any/binary[class_weighting][start][bf16][upscale_4][lr_1e-6][epoch_3]/20251222_042343/checkpoint_epoch1 dev

# 7B
# Librispeech
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen7b/librispeech/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_3][no_schedule][lr_1e-6].yaml outputs/qwen7b/librispeech/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_3][no_schedule][lr_1e-6]/20251222_042342/checkpoint_epoch3 dev
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen7b/librispeech/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_3][no_schedule][lr_1e-6].yaml outputs/qwen7b/librispeech/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_3][no_schedule][lr_1e-6]/20251222_042342/checkpoint_epoch2 dev
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen7b/librispeech/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_3][no_schedule][lr_1e-6].yaml outputs/qwen7b/librispeech/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_3][no_schedule][lr_1e-6]/20251222_042342/checkpoint_epoch1 dev

# after picking the best, epoch 3 does the best on dev/smooth_20ms_boxcar_broken/aux_correct_20ms

# Libricount
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen7b/libricount/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_6][no_schedule][lr_1e-6].yaml outputs/qwen7b/libricount/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_6][no_schedule][lr_1e-6]/20251222_042342/checkpoint_epoch6 dev
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen7b/libricount/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_6][no_schedule][lr_1e-6].yaml outputs/qwen7b/libricount/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_6][no_schedule][lr_1e-6]/20251222_042342/checkpoint_epoch5 dev
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen7b/libricount/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_6][no_schedule][lr_1e-6].yaml outputs/qwen7b/libricount/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_6][no_schedule][lr_1e-6]/20251222_042342/checkpoint_epoch4 dev
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen7b/libricount/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_6][no_schedule][lr_1e-6].yaml outputs/qwen7b/libricount/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_6][no_schedule][lr_1e-6]/20251222_042342/checkpoint_epoch3 dev
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen7b/libricount/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_6][no_schedule][lr_1e-6].yaml outputs/qwen7b/libricount/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_6][no_schedule][lr_1e-6]/20251222_042342/checkpoint_epoch2 dev
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen7b/libricount/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_6][no_schedule][lr_1e-6].yaml outputs/qwen7b/libricount/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_6][no_schedule][lr_1e-6]/20251222_042342/checkpoint_epoch1 dev

# Audioset
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen7b/audioset_humans_resampled/timestamp_any/binary[class_weighting][start][bf16][upscale_4][lr_1e-6][epoch_6].yaml outputs/qwen7b/audioset_humans_resampled/timestamp_any/binary[class_weighting][start][bf16][upscale_4][lr_1e-6][epoch_6]/20251222_054053/checkpoint_epoch6 dev
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen7b/audioset_humans_resampled/timestamp_any/binary[class_weighting][start][bf16][upscale_4][lr_1e-6][epoch_6].yaml outputs/qwen7b/audioset_humans_resampled/timestamp_any/binary[class_weighting][start][bf16][upscale_4][lr_1e-6][epoch_6]/20251222_054053/checkpoint_epoch5 dev
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen7b/audioset_humans_resampled/timestamp_any/binary[class_weighting][start][bf16][upscale_4][lr_1e-6][epoch_6].yaml outputs/qwen7b/audioset_humans_resampled/timestamp_any/binary[class_weighting][start][bf16][upscale_4][lr_1e-6][epoch_6]/20251222_054053/checkpoint_epoch4 dev
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen7b/audioset_humans_resampled/timestamp_any/binary[class_weighting][start][bf16][upscale_4][lr_1e-6][epoch_6].yaml outputs/qwen7b/audioset_humans_resampled/timestamp_any/binary[class_weighting][start][bf16][upscale_4][lr_1e-6][epoch_6]/20251222_054053/checkpoint_epoch3 dev
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen7b/audioset_humans_resampled/timestamp_any/binary[class_weighting][start][bf16][upscale_4][lr_1e-6][epoch_6].yaml outputs/qwen7b/audioset_humans_resampled/timestamp_any/binary[class_weighting][start][bf16][upscale_4][lr_1e-6][epoch_6]/20251222_054053/checkpoint_epoch2 dev
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen7b/audioset_humans_resampled/timestamp_any/binary[class_weighting][start][bf16][upscale_4][lr_1e-6][epoch_6].yaml outputs/qwen7b/audioset_humans_resampled/timestamp_any/binary[class_weighting][start][bf16][upscale_4][lr_1e-6][epoch_6]/20251222_054053/checkpoint_epoch1 dev

# ----------- LIBRISPEECH SWEEP -----------

# Tokens 3B, best 20ms
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/new_inference/token[start][bf16][lr_1e-6][epoch_3][no_schedule].yaml outputs/qwen3b/librispeech/timestamp_any/new_inference/token[start][bf16][lr_1e-6][epoch_3][no_schedule]/20251211_180507/model_epoch1.pt test

# Poisson 3B, best 40ms
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/new_inference/poisson[start][bias_-6][bf16][upscale_4][lr_1e-6][epoch_3][no_schedule].yaml outputs/qwen3b/librispeech/timestamp_any/new_inference/poisson[start][bias_-6][bf16][upscale_4][lr_1e-6][epoch_3][no_schedule]/20251210_233504/model_epoch1.pt test

# ABLATIONS
# Tokens <4, best 20ms
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_4].yaml outputs/qwen3b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_4]/20251221_040529/checkpoint_epoch3 test 4 8

# Tokens <8, best 20ms
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_8].yaml outputs/qwen3b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_8]/20251221_100430/checkpoint_epoch2 test 8 12

# Tokens <12, best 20ms
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_12].yaml outputs/qwen3b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_12]/20251221_112231/checkpoint_epoch3 test 12 16

# WE ALREADY EVALUATED CHECKPOINT 3 FOR BOTH OF THESE
# Tokens <16, best 20ms
# Tokens <20, best 20ms

# WE ALREADY HAVE <4, <8, <12, <20 for poisson
# Poisson <16, best 20ms
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_16].yaml outputs/qwen3b/librispeech/timestamp_any/ablation/poisson+bidirectional_audio[start][bias_-6][bf16][max_16]/20251224_222331/checkpoint_epoch3 test 16 20


# ----------- BERNOULLI TEST SWEEP -----------

# THESE ARE BASED ON THE TECHNICALLY BROKEN INFERENCE STRATEGIES

# 3B
# Librispeech
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_3][no_schedule][lr_1e-6].yaml outputs/qwen3b/librispeech/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_3][no_schedule][lr_1e-6]/20251221_235944/checkpoint_epoch3 test

# Libricount
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/libricount/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_3][no_schedule][lr_1e-6].yaml outputs/qwen3b/libricount/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_3][no_schedule][lr_1e-6]/20251222_014534/checkpoint_epoch3 test

# Audioset
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/audioset_humans_resampled/timestamp_any/binary[class_weighting][start][bf16][upscale_4][lr_1e-6][epoch_3].yaml outputs/qwen3b/audioset_humans_resampled/timestamp_any/binary[class_weighting][start][bf16][upscale_4][lr_1e-6][epoch_3]/20251222_042343/checkpoint_epoch3 test

# 7B
# Librispeech
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen7b/librispeech/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_3][no_schedule][lr_1e-6].yaml outputs/qwen7b/librispeech/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_3][no_schedule][lr_1e-6]/20251222_042342/checkpoint_epoch3 test

# Libricount
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen7b/libricount/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_6][no_schedule][lr_1e-6].yaml outputs/qwen7b/libricount/timestamp_any/new_inference/bernoulli[class_weighting][start][bf16][upscale_4][epoch_6][no_schedule][lr_1e-6]/20251222_042342/checkpoint_epoch6 test

# Audioset
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen7b/audioset_humans_resampled/timestamp_any/binary[class_weighting][start][bf16][upscale_4][lr_1e-6][epoch_6].yaml outputs/qwen7b/audioset_humans_resampled/timestamp_any/binary[class_weighting][start][bf16][upscale_4][lr_1e-6][epoch_6]/20251222_054053/checkpoint_epoch3 test

# Ablations
# < 4
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_4].yaml outputs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_4]/20251223_061132/checkpoint_epoch3 test 4 8 

# < 8
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_8].yaml outputs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_8]/20251223_061142/checkpoint_epoch3 test 8 12

# < 12
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_12].yaml outputs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_12]/20251223_061142/checkpoint_epoch3 test 12 16

# < 16
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_16].yaml outputs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_16]/20251223_084333/checkpoint_epoch2 test 16 20

# < 20
sbatch $EXCLUDE_NODES scripts/anvil/accelerate_1_gpu_bf16_eval.sh configs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_20].yaml outputs/qwen3b/librispeech/timestamp_any/ablation/bernoulli+bidirectional_audio+class_weighting[start][bf16][max_20]/20251223_084347/checkpoint_epoch3 test 20