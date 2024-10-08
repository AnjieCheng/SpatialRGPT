#!/bin/bash
MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# please clone depthanything
# git clone https://github.com/LiheYoung/Depth-Anything.git
# wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth
# place depth_anything_vitl14.pth under Depth-Anything/checkpoints
export DEPTH_ANYTHING_PATH="PATH_TO_DEPTHANYTHING"

CHUNKS=${#GPULIST[@]}

ANNOTATION_FILE=/PATH/SpatialRGPT-Bench_v1.json
OMNI3D_DATA_FOLDER=/PATH/omni3d/datasets/

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.eval_spatial \
        --model-path $MODEL_PATH \
        --annotation-file $ANNOTATION_FILE \
        --image-folder $OMNI3D_DATA_FOLDER \
        --answers-file ./eval_output/$CKPT/spatial/v1/answers/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode $CONV_MODE &
done

wait

output_file=./eval_output/$CKPT/spatial/v1/answers/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./eval_output/$CKPT/spatial/v1/answers/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done



python scripts/srgpt/eval/evaluate_spatial_with_gpt4.py $output_file
