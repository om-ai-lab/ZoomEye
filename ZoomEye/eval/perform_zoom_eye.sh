#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
MODEL_PATH=$1
ANNO_PATH=$2
BENCHMARK=$3

# The number of splits of the sub-images for each node
SPLIT_NUM=4



ANSWERS_DIR=ZoomEye/eval/answers/${BENCHMARK}/$(basename "$MODEL_PATH")
mkdir -p ${ANSWERS_DIR}

echo "MODEL_PATH: ${MODEL_PATH}"
echo "ANNO_PATH: ${ANNO_PATH}"
echo "BENCHMARK: ${BENCHMARK}"
echo "ANSWERS_DIR: ${ANSWERS_DIR}"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ZoomEye/eval/perform_zoom_eye.py \
        --model-path ${MODEL_PATH} \
        --answers-file ${ANSWERS_DIR}/${CHUNKS}_${IDX}.jsonl \
        --annotation_path ${ANNO_PATH} \
        --benchmark ${BENCHMARK} \
        --split-num ${SPLIT_NUM} \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait

output_file=${ANSWERS_DIR}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${ANSWERS_DIR}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done
