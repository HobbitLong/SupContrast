export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 main_supcon.py --batch_size 32 \
                      --learning_rate 0.5 \
                      --model densenet121 \
                      --match_type all \
                      --epoch 10 \
                      --dataset chexpert \
                      --temp 0.1 \
                      --cosine


