# A Quick Training Script Guide for the End-to-End Text Image Translation with Auxiliary Tasks.
# Please update the corresponding path and hyper-parameters before running the code in your own environment!
echo 'Please update the corresponding path and hyper-parameters before running the code in your own environment!'

code_path=${code_path}/E2E_TIT_With_MT/trainer.py
src_max_len=${max_length_of_source_language}
tgt_max_len=${max_length_of_source_language}
let img_width=${src_max_len}*4      # To make the sequence length of image features and text features consistent
model_path=${path_of_model_saving}
exp_name=${name_of_model_setting}   # Finally, the model is saved in ${model_path}/${exp_name}/
batch_size=${batch_size}
task_name=multi_task_ocr_mt
tit_weight=${loss_weight_of_tit_task}
ocr_weight=${loss_weight_of_ocr_task}
mt_weight=${loss_weight_of_mt_task}

# Path of End-to-end Text Image Translation Dataset | lmdb file.
train_path=${path_of_e2e_tit_train_dataset}
valid_path=${path_of_e2e_tit_valid_dataset}

# Path of Textual Machine Translation Dataset | txt file.
txt_train_src=${path_of_text_mt_train_dataset_source_language}
txt_train_tgt=${path_of_text_mt_train_dataset_target_language}

# Path of Vocabulary | txt file.
vocab_src=${path_of_source_language_vocabulary}
vocab_tgt=${path_of_target_language_vocabulary}

echo 'Remove Previous Model Folder.'
rm -rf ${model_path}/${exp_name}/

echo 'Start to train ...'
${python_path} ${code_path} \
--task ${task_name} \
--imgW ${img_width} --rgb \
--train_data ${train_path} \
--valid_data ${valid_path} \
--src_train_text ${txt_train_src} \
--tgt_train_text ${txt_train_tgt} \
--src_vocab ${vocab_src} --tgt_vocab ${vocab_tgt} \
--src_batch_max_length ${src_max_len} \
--tgt_batch_max_length ${tgt_max_len} \
--sensitive \
--batch_size ${batch_size} \
--saved_model ${model_path} --exp_name ${exp_name} \
--external_mt yes \
--TIT_Weight ${tit_weight} \
--OCR_Task --OCR_Weight ${ocr_weight} \
--MT_Task --MT_Weight ${mt_weight} \

echo 'Finished Training.'
