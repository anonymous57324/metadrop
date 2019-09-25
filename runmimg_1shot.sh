python main.py \
  --savedir './results/mimgnet_1shot' \
  --dataset 'mimgnet' \
  --mode 'train' \
  --gpu_num 0 \
  --metabatch 4 \
  --inner_lr 0.01 \
  --n_steps 5 \
  --way 5 \
  --shot 1 \
  --query 15 \
  --meta_lr 1e-4 \
  --n_test_mc_samp 1

python main.py \
  --savedir './results/mimgnet_1shot' \
  --dataset 'mimgnet' \
  --mode 'test' \
  --gpu_num 0 \
  --metabatch 1 \
  --inner_lr 0.01 \
  --n_steps 5 \
  --way 5 \
  --shot 1 \
  --query 15 \
  --meta_lr 1e-4 \
  --n_test_mc_samp 30
