GPU_NUM="1"
CFG="config/cfg_odvg.py"
DATASETS="config/datasets_mixed_odvg.json"
OUTPUT_DIR="/home/ec2-user/gdino/Open-GroundingDino/output/training"
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
 
# Change ``pretrain_model_path`` to use a different pretrain.
# (e.g. GroundingDINO pretrain, DINO pretrain, Swin Transformer pretrain.)
# If you don't want to use any pretrained model, just ignore this parameter.
 
python -m torch.distributed.launch  --nproc_per_node=${GPU_NUM} main.py \
        --output_dir ${OUTPUT_DIR} \
        -c ${CFG} \
        --datasets ${DATASETS}  \
        --pretrain_model_path /home/ec2-user/gdino/GroundingDINO/weights/groundingdino_swint_ogc.pth \
        --options text_encoder_type=/home/ec2-user/gdino/GroundingDINO/weights/bert-base-uncased use_coco_eval=False

python -m torch.distributed.launch --nproc_per_node=1 main.py \
  --output_dir /home/ec2-user/gdino/Open-GroundingDino/output/training \
  -c config/cfg_odvg.py \
  --datasets config/datasets_mixed_odvg.json \
  --pretrain_model_path /home/ec2-user/gdino/GroundingDINO/weights/groundingdino_swint_ogc.pth \
  --options text_encoder_type=/home/ec2-user/gdino/GroundingDINO/weights/bert-base-uncased use_coco_eval=False label_list="['LCAC','HUMVEE','LAV-25','lapan_uav_lsu','Osprey','m88','m1a1abrams','M777Howitzer','LCU','acv']"