#!/bin/sh
export MASTER_PORT=6055
echo MASTER_PORT=${MASTER_PORT}

export PYTHONPATH=$(pwd):$PYTHONPATH

CURDIR=$(cd $(dirname $0); pwd)
echo 'The work dir is: ' $CURDIR

DATASET=brats
MODE=test
GPUS=1

if [ -z "$1" ]; then
   GPUS=1
fi

echo $DATASET $MODE $GPUS

# ----------------- Wavelet -----------
if [[ $MODE == train ]]; then
	echo "==> Training WaveDiff"

  if [[ $DATASET == brats ]]; then
    python train_wddgan.py --dataset brats --image_size 128 --exp wddgan_brats_3d --num_channels 24 --num_channels_dae 64 --ch_mult 1 2 2 2 4 --num_timesteps 8 \
      --num_res_blocks 4 --batch_size 3 --num_epoch 1000000 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
      --z_emb_dim 256 --lr_d 1e-5 --lr_g 2e-5 --datadir /absolute/path/to/training/data/ \
      --master_port $MASTER_PORT --num_process_per_node $GPUS \
      --current_resolution 64 \
      --net_type wavelet --beta_min 0.1 --beta_max 20.0 --local_rank 0 \

	fi

else
	echo "==> Testing WaveDiff"

	if [[ $DATASET == brats ]]; then \
		python test_wddgan.py --dataset brats --image_size 128 --exp wddgan_brats_3d --num_channels 24 --num_channels_dae 64 --ch_mult 1 2 2 2 4 --num_timesteps 8 \
      --num_res_blocks 4 --batch_size 1 --embedding_type positional \
      --z_emb_dim 256 \
      --current_resolution 64 --attn_resolution 16 \
      --net_type wavelet --checkpoint_path /path/to/trained_model.pth \
		  --datadir /absolute/path/to/testing/data/ --beta_min 0.1 --beta_max 20.0 \

	fi

fi
