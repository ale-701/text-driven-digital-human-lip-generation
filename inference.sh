conda activate fish-speech
cd /home/zxy/fish-speech
python tools/llama/generate.py --text "Your text here" --checkpoint-path "checkpoints/fish-speech-1.5" --output-dir output
python fish_speech/models/vqgan/inference.py -i "output/codes_0.npy" --checkpoint-path "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"  --output-path 'output/sound.wav'