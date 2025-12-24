# EXCLUDE_NODES="--exclude=h012"
# qwen3b
# sbatch scripts/hyak/base_eval.sh Qwen/Qwen2.5-Omni-3B gilkeyio/librispeech-alignments test SINGLE_WORD_TIMESTAMP_ANY
# sbatch scripts/hyak/base_eval.sh Qwen/Qwen2.5-Omni-3B enyoukai/libricount-timings test SINGLE_WORD_TIMESTAMP_ANY
# sbatch scripts/hyak/base_eval.sh Qwen/Qwen2.5-Omni-3B enyoukai/audioset-humans-reprocessed test SINGLE_WORD_TIMESTAMP_ANY

# # qwen7b
# sbatch scripts/hyak/base_eval.sh Qwen/Qwen2.5-Omni-7B gilkeyio/librispeech-alignments test SINGLE_WORD_TIMESTAMP_ANY
# sbatch scripts/hyak/base_eval.sh Qwen/Qwen2.5-Omni-7B enyoukai/libricount-timings test SINGLE_WORD_TIMESTAMP_ANY
# sbatch scripts/hyak/base_eval.sh Qwen/Qwen2.5-Omni-7B enyoukai/audioset-humans-reprocessed test SINGLE_WORD_TIMESTAMP_ANY

# voxtral 3b
sbatch scripts/hyak/base_eval.sh mistralai/Voxtral-Mini-3B-2507 gilkeyio/librispeech-alignments test SINGLE_WORD_TIMESTAMP_ANY
sbatch scripts/hyak/base_eval.sh mistralai/Voxtral-Mini-3B-2507 enyoukai/libricount-timings test SINGLE_WORD_TIMESTAMP_ANY
sbatch scripts/hyak/base_eval.sh mistralai/Voxtral-Mini-3B-2507 enyoukai/audioset-humans-reprocessed test SINGLE_WORD_TIMESTAMP_ANY

# voxtral 24b
sbatch scripts/hyak/base_eval_a100.sh mistralai/Voxtral-Small-24B-2507 gilkeyio/librispeech-alignments test SINGLE_WORD_TIMESTAMP_ANY
sbatch scripts/hyak/base_eval_a100.sh mistralai/Voxtral-Small-24B-2507 enyoukai/libricount-timings test SINGLE_WORD_TIMESTAMP_ANY
sbatch scripts/hyak/base_eval_a100.sh mistralai/Voxtral-Small-24B-2507 enyoukai/audioset-humans-reprocessed test SINGLE_WORD_TIMESTAMP_ANY

# audio flamingo 3
# sbatch scripts/hyak/base_eval.sh nvidia/audio-flamingo-3-hf gilkeyio/librispeech-alignments test SINGLE_WORD_TIMESTAMP_ANY
# sbatch scripts/hyak/base_eval.sh nvidia/audio-flamingo-3-hf enyoukai/libricount-timings test SINGLE_WORD_TIMESTAMP_ANY
# sbatch scripts/hyak/base_eval.sh nvidia/audio-flamingo-3-hf enyoukai/audioset-humans-reprocessed test SINGLE_WORD_TIMESTAMP_ANY
