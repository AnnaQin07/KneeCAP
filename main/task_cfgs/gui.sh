mkdir -p log


checkpoint_path=$1

python main.py --task "gui" \
               --config "configs/inference.yaml" 2>&1 \
               --checkpoint_path ${checkpoint_path}