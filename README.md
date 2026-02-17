# Setup

## Environment Setup

1. First, modify the `prefix` argument in `environment.yaml` to point to your desired directory (or current directory):
   ```yaml
   prefix: /path/to/your/desired/environment/location
   ```

2. Create the environment from the YAML file:
   ```bash
   conda env create -f environment.yaml
   ```

3. Activate the environment:
   ```bash
   conda activate ./env # or whatever you set prefix to
   ```

## Download NLTK Data

Run the following Python commands to download required NLTK data:

```python
import nltk
nltk.download('stopwords')
```

## Datasets

The datasets can be found on HuggingFace

1. **LibriSpeech**  
   [https://huggingface.co/datasets/gilkeyio/librispeech-alignments](https://huggingface.co/datasets/gilkeyio/librispeech-alignments)  

2. **LibriCount**  
   [https://huggingface.co/datasets/enyoukai/libricount-timings](https://huggingface.co/datasets/enyoukai/libricount-timings)  

3. **AudioSet**  
   [https://huggingface.co/datasets/enyoukai/audioset-humans-reprocessed](https://huggingface.co/datasets/enyoukai/audioset-humans-reprocessed)  

## Running experiments

### Single timestamps

To train on single timestamps, run

```
accelerate launch --config_file ACCELERATE_CONFIG_PATH run.py --config CONFIG_PATH
```

Evaluate on single timestamps with the following command:

```
accelerate launch --config_file ACCELERATE_CONFIG_PATH run.py --config CONFIG_PATH --load-checkpoint CHECKPOINT_PATH --eval-only
```

To evaluate on the dev split, add `--dev`. By default, `--eval-only` evaluates on test

To run ablations, use the arguments `--eval-min-time` and `--eval-max-time` to set the bounds of which timestamps should be sampled. `--eval-max-time` can also be omitted to not set a max-time limit.

For example, to train and evaluate Qwen 2.5 Omni 3B on Librispeech on timestamps up to 8 seconds while using 4 GPUs, run the following command:

```
# evaluate
accelerate launch --config_file accelerate_configs/4_gpu_bf16.yaml run.py --config configs/qwen3b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_8].yaml

# evaluate
accelerate launch --config_file accelerate_configs/4_gpu_bf16.yaml run.py --config configs/qwen3b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_8].yaml --load-checkpoint outputs/qwen3b/librispeech/timestamp_any/ablation/token+bidirectional_audio[start][bf16][max_8]/20251221_100430/checkpoint_epoch3 --eval-only

```

### Multi timestamps
Same as above, except run `ga_run.py` instead of `run.py`

Training and evaluating Qwen 2.5 Omni 3B on Librispeech with 2 GPUs:

```
# train
accelerate launch --config_file accelerate_configs/2_gpu_bf16.yaml ga_run.py --config configs/qwen3b/librispeech/timestamp_all/token[start][bf16][lr_1e-6][epoch_3][no_schedule].yaml

# evaluate
accelerate launch --config_file accelerate_configs/2_gpu_bf16.yaml ga_run.py --config configs/qwen3b/librispeech/timestamp_all/token[start][bf16][lr_1e-6][epoch_3][no_schedule].yaml --load-checkpoint outputs/qwen3b/librispeech/timestamp_all/token[start][bf16][lr_1e-6][epoch_3][no_schedule]/20251230_230106/checkpoint_epoch3 --eval-only 
```

By defualt, all checkpoints will be saved under outputs/. 

### Writing new configs

The configs/ folder has example of existing configs.
