### Required and Saved Configuration

#### required data folder configuration
```bash
├─── data_dir
│   ├─── 200x_256 (magnification and patch size)
│       ├─── {slide_id}_{x}_{y}_input.jpg
│       ├─── {slide_id}_{x}_{y}_label.png
│       ├─── ...
```

#### model saving folders
```bash
├─── model_dir
│   ├─── 1-fold (test fold for 5-fold cv)
│       ├─── log (via tensorboard)
│           ├─── train
│           ├─── valid
│       ├───checkpoint
│           ├─── model_epoch1.pth
│           ├─── ...
```

### Run Codes

#### train
```terminal
python3 train.py --fold 1 \
                 --data_dir '/data' \
                 --model_dir '/model/UNet_B' \
                 --model_arch 'UNet_B' \
                 --optim 'Adam' --lr 1e-3 --loss 'BCElogit' \
                 --n_epoch 200 --batch_size 128 \
                 --local_rank 0 1 2 3 4 5 6 7 \
                 --log_img 1

python3 train.py --fold 1 \
                 --data_dir '/data \
                 --model_dir '/model/SUNet_B' \
                 --model_arch 'UNet_B' \
                 --selective 1 --s_lamb 2 \
                 --optim 'Adam' --lr 1e-3 --loss 'BCElogit' \
                 --n_epoch 200 --batch_size 128 \
                 --local_rank 0 1 2 3 4 5 6 7 \
                 --log_img 1
```

#### eval
```terminal
python3 eval.py --fold 1 \
                --data_dir '/data' 
                --model_dir '/model/UNet_B' \
                --model_arch 'UNet_B' \ 
                --batch_size 128 \
                --local_rank 0 1 2 3 4 5 6 7 

python3 eval.py --fold 1 \
                --data_dir '/data' 
                --model_dir '/model/SUNet_B' \
                --model_arch 'UNet_B' \
                --selective 1 \
                --select_eval 1 \ 
                --batch_size 128 \
                --local_rank 0 1 2 3 4 5 6 7 
                      
```
if you don't assign select_eval 1, assessment witout selection will be conducted

you can use a certain single gpu and multiple-gpus including gpu:0 
ex)
--local_rank 0 
--local_rank 2
--local_rank 0 1 2 3

you have to assign select_eval as 1 in order to conduct in-coverage assessment
--select_eval 1

you have to assign gpu:0 to use DataParallel 
(X) --local_rank 1 2 3

### Qunatitative Results

||Accuracy|Recall|Precision|F1 Score|IoU (Benign)|IoU (Tumor)|mIoU|
|:-:|:-:|:-|:-:|:-:|:-:|:-:|:-:|
|UNet|0.9329|0.8615|0.9258|0.8925|0.9070|0.8059|0.8565|
|SelectiveUNet|0.9839|0.9654|0.9777|0.9715|0.9778|0.9446|0.9612|
|SelectiveUNet(w/o selection)| 0.9166|0.7946|0.9381|0.8604|0.8878|0.7550|0.8214|

