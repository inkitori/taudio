conda create -n taudio python=3.12.11
conda activate taudio
pip install -r requirements.txt
conda env config vars set HF_HOME=".cache/huggingface" # need to reactivate environment
conda install anaconda::ffmpeg 