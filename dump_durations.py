from dataset import create_adapter as create_dataset_adapter
from dataset import infer_adapter_from_repository
from tasks.timestamp_single_any import SingleTimestampAnyTask

import csv

REPOSITORY = "gilkeyio/librispeech-alignments"
output_file = f"dumped_durations/{REPOSITORY.split('/')[1]}.csv"

task = SingleTimestampAnyTask()
dataset_name = infer_adapter_from_repository(REPOSITORY)
ds_adapter = create_dataset_adapter(
    dataset_name,
    sampling_rate=16000,
    repository=REPOSITORY,
    left_padding=0,
    key=task,
)

split = ds_adapter.load_split('test')
split = task.select_indices(split, ds_adapter, 'test')

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    # Write Header
    writer.writerow(["gt_timestamp", "duration_seconds"])
    
    for i, ex in enumerate(split):
        if task.skip_example(ex, ds_adapter):
            continue
            
        events = list(ds_adapter.get_events(ex))
        event = task._choose_event(events=events, ds_adapter=ds_adapter, apply_fallback=False, example=ex)  
        gt = ds_adapter.get_target_seconds(event, task.key)

        audio_duration = ex['audio']['array'].size / ex['audio']['sampling_rate']
        writer.writerow([gt, audio_duration])