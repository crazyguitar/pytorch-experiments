.PHONY: all download run

all: run

run:
	torchrun --nnodes 1 --nproc_per_node 8 T5_training.py --epochs 10

download:
	sh download_dataset.sh
