python tools/inference_on_a_image.py \
  -c tools/GroundingDINO_SwinT_OGC.py \
  -p /home/ec2-user/gdino/GroundingDINO/weights/groundingdino_swint_ogc.pth \
  -i ./figs/dog_cat.jpeg \
  -t "dog" \
  -o output