python coco2odvg_flex2.py \
-i /home/ec2-user/data/data/training-labels/macro/xtech_coco_annotations.json \
--train-output /home/ec2-user/data/data/training-labels/macro/xtech_train_odvg.jsonl \
--val-output /home/ec2-user/data/data/training-labels/macro/xtech_val_coco.json \
--save-label-map /home/ec2-user/data/data/training-labels/macro/xtech_label_mapping.json \
--val-ratio 0.2 \
--seed 42