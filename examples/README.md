### Running MAML

Usage
```bash
python3 examples/run_maml_glue.py  --model_name_or_path bert-base-uncased  --do_train  --do_eval --max_seq_length 128   --per_device_train_batch_size 1  --learning_rate 2e-5  --output_dir /home/nlp/experiments/fluence_exp/   --overwrite_output_dir --per_device_eval_batch_size 4096 --data_dir $GLUE_DIR --train_task mrpc --eval_task sst-2 --save_steps=10000 --num_train_epochs=1 --output_file_name check --eval_method every_2
```
