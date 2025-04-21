#!/bin/bash

# Check if the user provided input text
if [ -z "$1" ]; then
    echo "Usage: $0 <text>"
    exit 1
fi

# Define variables
INPUT_TEXT="$1"
OUTPUT_DIR=$(pwd)
ORIGINAL_DIR=$(pwd)
VIDEO="$2"
REF_AUDIO="$3"
REF_TEXT="$4"
TTS_ENV_NAME="fish-speech"
TTS_PROJECT_DIR="/home/zxy/fish-speech"

LS_ENV_NAME="latentsync"
LS_PROJECT_DIR="/home/zxy/LatentSync-main"

ESR_ENV_NAME="esrgan"
ESR_PROJECT_DIR="/home/zxy/Real-ESRGAN"


rm video_out_out.mp4

echo "Using TTS environment: $TTS_ENV_NAME"

# Use conda run instead of activating
conda run -n $TTS_ENV_NAME python $TTS_PROJECT_DIR/fish_speech/models/vqgan/inference.py -i "$REF_AUDIO" --checkpoint-path "$TTS_PROJECT_DIR/checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth" --output-path "$OUTPUT_DIR/fake.wav"
conda run -n $TTS_ENV_NAME python $TTS_PROJECT_DIR/fish_speech/models/text2semantic/inference.py \
    --text "$INPUT_TEXT" \
    --prompt-text "$REF_TEXT" \
    --prompt-tokens "$OUTPUT_DIR/fake.npy" \
    --checkpoint-path "$TTS_PROJECT_DIR/checkpoints/fish-speech-1.5" \
    --num-samples 2
# # conda run -n $TTS_ENV_NAME python $TTS_PROJECT_DIR/tools/llama/generate.py --text "$INPUT_TEXT" --checkpoint-path "$TTS_PROJECT_DIR/checkpoints/fish-speech-1.5" --output-dir "$OUTPUT_DIR"

conda run -n $TTS_ENV_NAME python $TTS_PROJECT_DIR/fish_speech/models/vqgan/inference.py -i "$OUTPUT_DIR/temp/codes_0.npy" --checkpoint-path "$TTS_PROJECT_DIR/checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth" --output-path "$OUTPUT_DIR/sound.wav"

cd "$LS_PROJECT_DIR"

conda run -n $LS_ENV_NAME python -m scripts.inference \
    --unet_config_path "configs/unet/second_stage.yaml" \
    --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
    --guidance_scale 1.5 \
    --video_path "$VIDEO" \
    --audio_path "$OUTPUT_DIR/sound.wav" \
    --video_out_path "$OUTPUT_DIR/video_out.mp4"



cd "$ESR_PROJECT_DIR"
conda run -n $ESR_ENV_NAME python inference_realesrgan_video.py -n RealESRGAN_x4plus -i "$OUTPUT_DIR/video_out.mp4"  -s 2 -o "$OUTPUT_DIR"

cd "$ORIGINAL_DIR"