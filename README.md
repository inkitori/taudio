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

1. **LibriSpeech Alignments**  
   [https://huggingface.co/datasets/gilkeyio/librispeech-alignments](https://huggingface.co/datasets/gilkeyio/librispeech-alignments)  

2. **LibriCount Timings**  
   [https://huggingface.co/datasets/enyoukai/libricount-timings](https://huggingface.co/datasets/enyoukai/libricount-timings)  

3. **AudioSet Humans (Reprocessed)**  
   [https://huggingface.co/datasets/enyoukai/audioset-humans-reprocessed](https://huggingface.co/datasets/enyoukai/audioset-humans-reprocessed)  
