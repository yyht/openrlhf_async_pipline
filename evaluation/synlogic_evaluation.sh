

export TOKENIZERS_PARALLELISM=false
top_p=${TOP_P:-1}

echo "TOP_P $top_p"


python3 -u synlogic_evaluation.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_dir ${DATA_DIR} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT_TYPE} \
    --input_key ${INPUT_KEY} \
    --answer_key ${ANSWER_KEY} \
    --seed 0 \
    --temperature ${TEMPERATURE} \
    --n_sampling ${N_SAMPLING} \
    --top_p ${top_p} \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    # --overwrite \