# Setup

## Environment Setup

Create and activate the conda environment:

```bash
conda create -n taudio python=3.12.11
conda activate taudio
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Configure Hugging Face Cache

```bash
mkdir -p .cache/huggingface
conda env config vars set HF_HOME=".cache/huggingface"
# Note: You need to reactivate the environment after setting this variable
conda deactivate
conda activate taudio
```

## Install FFmpeg

```bash
conda install anaconda::ffmpeg
```

## Download NLTK Data

Run the following Python commands to download required NLTK data:

```python
import nltk
nltk.download('stopwords')
```