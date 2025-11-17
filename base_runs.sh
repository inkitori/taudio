EXCLUDE_NODES="--exclude=h012"
# qwen3b
sbatch $EXCLUDE_NODES scripts/anvil/base_eval.sh Qwen/Qwen2.5-Omni-3B gilkeyio/librispeech-alignments test SINGLE_WORD_TIMESTAMP_ANY
sbatch $EXCLUDE_NODES scripts/anvil/base_eval.sh Qwen/Qwen2.5-Omni-3B enyoukai/libricount-timings test SINGLE_WORD_TIMESTAMP_ANY
sbatch $EXCLUDE_NODES scripts/anvil/base_eval.sh Qwen/Qwen2.5-Omni-3B enyoukai/AudioSet-Strong-Human-Sounds test SINGLE_WORD_TIMESTAMP_ANY

# # qwen7b
sbatch $EXCLUDE_NODES scripts/anvil/base_eval.sh Qwen/Qwen2.5-Omni-7B gilkeyio/librispeech-alignments test SINGLE_WORD_TIMESTAMP_ANY
sbatch $EXCLUDE_NODES scripts/anvil/base_eval.sh Qwen/Qwen2.5-Omni-7B enyoukai/libricount-timings test SINGLE_WORD_TIMESTAMP_ANY
sbatch $EXCLUDE_NODES scripts/anvil/base_eval.sh Qwen/Qwen2.5-Omni-7B enyoukai/AudioSet-Strong-Human-Sounds test SINGLE_WORD_TIMESTAMP_ANY

# # voxtral 3b
# sbatch $EXCLUDE_NODES scripts/anvil/base_eval.sh mistralai/Voxtral-Mini-3B-2507 gilkeyio/librispeech-alignments test
# sbatch $EXCLUDE_NODES scripts/anvil/base_eval.sh mistralai/Voxtral-Mini-3B-2507 enyoukai/libricount-timings test
# sbatch $EXCLUDE_NODES scripts/anvil/base_eval.sh mistralai/Voxtral-Mini-3B-2507 enyoukai/AudioSet-Strong-Human-Sounds test


# # voxtral 24b
# sbatch $EXCLUDE_NODES scripts/anvil/base_eval.sh mistralai/Voxtral-Small-24B-2507 gilkeyio/librispeech-alignments test
# sbatch $EXCLUDE_NODES scripts/anvil/base_eval.sh mistralai/Voxtral-Small-24B-2507 enyoukai/libricount-timings test
# sbatch $EXCLUDE_NODES scripts/anvil/base_eval.sh mistralai/Voxtral-Small-24B-2507 enyoukai/AudioSet-Strong-Human-Sounds test
