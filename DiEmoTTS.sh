export PYTHONPATH=.
DEVICE=0;

# data
CONFIG="configs/exp/DINO_FS2.yaml";
binary_data_dir="/workspace/none/hd0/dataset/binary/DINO_ESD"

# code 
task_class="tasks.tts.DiEmoTTS.ExpressiveFS2Task";
p_dataset_class="tasks.tts.dataset_utils.DINO_5crop_FS_cluster";
up_dataset_class="tasks.tts.dataset_utils.DINO_infer";
model_class="models.tts.DiEmoTTS.ExpressiveFS2";

# eval
GT_DIR="/workspace/none/hd0/dataset/ESD_all_test_cs";
SCRIPT_DIR="/workspace/none/hd0/dataset/scripts";
TGT_SPK_DIR="/workspace/none/hd0/dataset/ESD_tgt/"

run_infer() {
    local MODEL_NAME=$1
    for config_suffix in "train" "p_0013" "up_0013" "p_0019" "up_0019"; do
        local FINAL_MODEL_NAME=${MODEL_NAME}_${config_suffix}
        local GEN_DIR=/workspace/none/hd0/out/DiEmoTTS/${MODEL_NAME}/generated_160000_${config_suffix}/wavs

        if [ "$config_suffix" == "train" ]; then
            echo "config_suffix: $config_suffix"
            local HPARAMS="binary_data_dir=$binary_data_dir,task_cls=$task_class,dataset_cls=$p_dataset_class,model_cls=$model_class,gen_dir_name=$config_suffix"

            # Train
            CUDA_VISIBLE_DEVICES=$DEVICE python tasks/run.py \
                --config $CONFIG \
                --exp_name $MODEL_NAME \
                --hparams=$HPARAMS \
                --reset

            # Infer
            CUDA_VISIBLE_DEVICES=$DEVICE python tasks/run.py \
                --config $CONFIG \
                --exp_name $MODEL_NAME \
                --infer \
                --hparams=$HPARAMS \
                --reset

        elif [ "$config_suffix" == "p_0013" ]; then
            echo "config_suffix: $config_suffix"
            local SPK_ID_NAME="0013"
            local GEN_DIR_NAME="P"
            local HPARAMS="binary_data_dir=$binary_data_dir,task_cls=$task_class,dataset_cls=$p_dataset_class,model_cls=$model_class,gen_dir_name=$GEN_DIR_NAME,spk_name=$SPK_ID_NAME"
            # Infer
            CUDA_VISIBLE_DEVICES=$DEVICE python tasks/run.py \
                --config $CONFIG \
                --exp_name $MODEL_NAME \
                --infer \
                --hparams=$HPARAMS \
                --reset

        elif [ "$config_suffix" == "up_0013" ]; then
            echo "config_suffix: $config_suffix"
            local SPK_ID_NAME="0013"
            local GEN_DIR_NAME="UP"
            local HPARAMS="binary_data_dir=$binary_data_dir,task_cls=$task_class,dataset_cls=$up_dataset_class,model_cls=$model_class,gen_dir_name=$GEN_DIR_NAME,spk_name=$SPK_ID_NAME"
            # Infer
            CUDA_VISIBLE_DEVICES=$DEVICE python tasks/run.py \
                --config $CONFIG \
                --exp_name $MODEL_NAME \
                --infer \
                --hparams=$HPARAMS \
                --reset

        elif [ "$config_suffix" == "p_0019" ]; then
            echo "config_suffix: $config_suffix"
            local SPK_ID_NAME="0019"
            local GEN_DIR_NAME="P"
            local FIANL_GEN_DIR=/workspace/none/hd0/out/DiEmoTTS/${MODEL_NAME}/generated_160000_P/wavs
            local HPARAMS="binary_data_dir=$binary_data_dir,task_cls=$task_class,dataset_cls=$p_dataset_class,model_cls=$model_class,gen_dir_name=$GEN_DIR_NAME,spk_name=$SPK_ID_NAME"
            # Infer
            CUDA_VISIBLE_DEVICES=$DEVICE python tasks/run.py \
                --config $CONFIG \
                --exp_name $MODEL_NAME \
                --infer \
                --hparams=$HPARAMS \
                --reset

        elif [ "$config_suffix" == "up_0019" ]; then
            echo "config_suffix: $config_suffix"
            local SPK_ID_NAME="0019"
            local GEN_DIR_NAME="UP"
            local FIANL_GEN_DIR=/workspace/none/hd0/out/DiEmoTTS/${MODEL_NAME}/generated_160000_UP/wavs
            local HPARAMS="binary_data_dir=$binary_data_dir,task_cls=$task_class,dataset_cls=$up_dataset_class,model_cls=$model_class,gen_dir_name=$GEN_DIR_NAME,spk_name=$SPK_ID_NAME"
            # Infer
            CUDA_VISIBLE_DEVICES=$DEVICE python tasks/run.py \
                --config $CONFIG \
                --exp_name $MODEL_NAME \
                --infer \
                --hparams=$HPARAMS \
                --reset

        fi
    done
}

#########################
#   Run for the model   #
#########################
run_infer "DiEmoTTS_final"