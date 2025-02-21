# 96
python -u main.py --data ETTh1 --input_len 96  --pred_len 192,336,720 --encoder_layer 3 --layer_stack 2 --MODWT_level 3 --patch_size 6 --d_model 8 --augmentation_len 48 --augmentation_ratio 0.5 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 10 --itr 5 --train --patience 2 --decay 0.5

python -u main.py --data ETTh2 --input_len 96  --pred_len 192,336,720 --encoder_layer 3 --layer_stack 2 --MODWT_level 3 --patch_size 6 --d_model 8 --augmentation_len 48 --augmentation_ratio 0.5 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 10 --itr 5 --train --patience 2 --decay 0.5

python -u main.py --data ETTm1 --input_len 96  --pred_len 192,336,720 --encoder_layer 3 --layer_stack 2 --MODWT_level 3 --patch_size 6 --d_model 32 --augmentation_len 48 --augmentation_ratio 0.5 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 10 --itr 5 --train --patience 2 --decay 0.5

python -u main.py --data ETTm2 --input_len 96  --pred_len 192,336,720 --encoder_layer 3 --layer_stack 2 --MODWT_level 3 --patch_size 6 --d_model 32 --augmentation_len 48 --augmentation_ratio 0.5 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 10 --itr 5 --train --patience 2 --decay 0.5

python -u main.py --data ECL --input_len 96  --pred_len 192,336,720 --encoder_layer 3 --layer_stack 2 --MODWT_level 3 --patch_size 6 --d_model 32 --augmentation_len 48 --augmentation_ratio 0.5 --learning_rate 0.0001 --dropout 0.1 --batch_size 4 --train_epochs 10 --itr 5 --train --decoder_IN --patience 2 --decay 0.5

python -u main.py --data Traffic --input_len 96  --pred_len 192,336,720 --encoder_layer 3 --layer_stack 2 --MODWT_level 3 --patch_size 6 --d_model 32 --augmentation_len 48 --augmentation_ratio 0.5 --learning_rate 0.0001 --dropout 0.1 --batch_size 4 --train_epochs 10 --itr 5 --train --patience 2 --decay 0.5

python -u main.py --data weather --input_len 96  --pred_len 192,336,720 --encoder_layer 3 --layer_stack 2 --MODWT_level 3 --patch_size 6 --d_model 32 --augmentation_len 48 --augmentation_ratio 0.5 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 10 --itr 5 --train  --decoder_IN --patience 2 --decay 0.5

python -u main.py --data Solar --input_len 96  --pred_len 192,336,720 --encoder_layer 3 --layer_stack 2 --MODWT_level 3 --patch_size 6 --d_model 32 --augmentation_len 48 --augmentation_ratio 0.5 --learning_rate 0.0001 --dropout 0.1 --batch_size 4 --train_epochs 10 --itr 5 --train --decoder_IN --patience 2 --decay 0.5

python -u main.py --data PEMS03 --input_len 96  --pred_len 192,336,720 --encoder_layer 3 --layer_stack 2 --MODWT_level 3 --patch_size 6 --d_model 32 --augmentation_len 48 --augmentation_ratio 0.5 --learning_rate 0.0001 --dropout 0.1 --batch_size 4 --train_epochs 10 --itr 5 --train --decoder_IN --patience 2 --decay 0.5

python -u main.py --data PEMS04 --input_len 96  --pred_len 192,336,720 --encoder_layer 3 --layer_stack 2 --MODWT_level 3 --patch_size 6 --d_model 32 --augmentation_len 48 --augmentation_ratio 0.5 --learning_rate 0.0001 --dropout 0.1 --batch_size 4 --train_epochs 10 --itr 5 --train --decoder_IN --patience 2 --decay 0.5

python -u main.py --data PEMS07 --input_len 96  --pred_len 192,336,720 --encoder_layer 3 --layer_stack 2 --MODWT_level 3 --patch_size 6 --d_model 32 --augmentation_len 48 --augmentation_ratio 0.5 --learning_rate 0.0001 --dropout 0.1 --batch_size 4 --train_epochs 10 --itr 5 --train --decoder_IN --patience 2 --decay 0.5

python -u main.py --data PEMS08 --input_len 96  --pred_len 192,336,720 --encoder_layer 3 --layer_stack 2 --MODWT_level 3 --patch_size 6 --d_model 32 --augmentation_len 48 --augmentation_ratio 0.5 --learning_rate 0.0001 --dropout 0.1 --batch_size 4 --train_epochs 10 --itr 5 --train --decoder_IN --patience 2 --decay 0.5

python -u main.py --data PEMSBay --input_len 96  --pred_len 192,336,720 --encoder_layer 3 --layer_stack 2 --MODWT_level 3 --patch_size 6 --d_model 32 --augmentation_len 48 --augmentation_ratio 0.5 --learning_rate 0.0001 --dropout 0.1 --batch_size 4 --train_epochs 10 --itr 5 --train --decoder_IN --patience 2 --decay 0.5

# 336
python -u main.py --data ETTh1 --input_len 336  --pred_len 192,336,720 --encoder_layer 3 --layer_stack 2 --MODWT_level 3 --patch_size 6 --d_model 8 --augmentation_len 48 --augmentation_ratio 0.5 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 10 --itr 5 --train --patience 2 --decay 0.5

python -u main.py --data weather --input_len 336  --pred_len 192,336,720 --encoder_layer 3 --layer_stack 2 --MODWT_level 3 --patch_size 6 --d_model 32 --augmentation_len 48 --augmentation_ratio 0.5 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 10 --itr 5 --train  --decoder_IN --patience 2 --decay 0.5

python -u main.py --data ECL --input_len 336  --pred_len 192,336,720 --encoder_layer 3 --layer_stack 2 --MODWT_level 3 --patch_size 6 --d_model 32 --augmentation_len 48 --augmentation_ratio 0.5 --learning_rate 0.0001 --dropout 0.1 --batch_size 4 --train_epochs 10 --itr 5 --train --decoder_IN --patience 2 --decay 0.5

python -u main.py --data Traffic --input_len 336  --pred_len 192,336,720 --encoder_layer 3 --layer_stack 2 --MODWT_level 3 --patch_size 6 --d_model 32 --augmentation_len 48 --augmentation_ratio 0.5 --learning_rate 0.0001 --dropout 0.1 --batch_size 4 --train_epochs 10 --itr 5 --train --patience 2 --decay 0.5
