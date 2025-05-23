import argparse
import jsonlines
from tqdm import tqdm
import json
from pycocotools.coco import COCO
from collections import defaultdict
import random
import copy

def coco_to_xyxy(bbox):
    """Convert COCO bbox format [x, y, width, height] to [x1, y1, x2, y2]"""
    x, y, width, height = bbox
    x1 = round(x, 2) 
    y1 = round(y, 2)
    x2 = round(x + width, 2)
    y2 = round(y + height, 2)
    return [x1, y1, x2, y2]

def create_label_mapping(coco, start_from_zero=True):
    """Create a mapping from original category IDs to sequential labels"""
    cats = coco.loadCats(coco.getCatIds())
    
    # Sort by category ID to ensure consistent mapping
    cats = sorted(cats, key=lambda x: x['id'])
    
    id_to_name = {cat['id']: cat['name'] for cat in cats}
    
    if start_from_zero:
        # Create sequential mapping starting from 0
        id_to_label = {cat['id']: idx for idx, cat in enumerate(cats)}
    else:
        # Keep original IDs
        id_to_label = {cat['id']: cat['id'] for cat in cats}
    
    print(f"Found {len(cats)} categories:")
    for cat in cats:
        original_id = cat['id']
        new_label = id_to_label[original_id]
        print(f"  {original_id} -> {new_label}: {cat['name']}")
    
    return id_to_label, id_to_name

def dump_label_map(id_to_label, id_to_name, output="./label_map.json"):
    """Save the label mapping to a JSON file"""
    label_map = {}
    for orig_id, new_label in id_to_label.items():
        label_map[new_label] = id_to_name[orig_id]
    
    with open(output, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Label mapping saved to {output}")

def stratified_split(coco, val_ratio=0.2, seed=42):
    """
    Perform stratified train-val split ensuring all categories are in both splits
    """
    random.seed(seed)
    
    # Group images by categories they contain
    category_to_images = defaultdict(set)
    
    for img_id, img_info in coco.imgs.items():
        ann_ids = coco.getAnnIds(imgIds=img_id)
        categories_in_image = set()
        for ann_id in ann_ids:
            ann = coco.anns[ann_id]
            categories_in_image.add(ann['category_id'])
        
        # Add this image to all categories it contains
        for cat_id in categories_in_image:
            category_to_images[cat_id].add(img_id)
    
    all_categories = list(category_to_images.keys())
    all_images = set(coco.imgs.keys())
    
    print(f"Total images: {len(all_images)}")
    print(f"Categories: {len(all_categories)}")
    
    # Ensure each category has at least one image in validation
    val_images = set()
    train_images = set()
    
    # First, ensure each category has representation in validation
    for cat_id in all_categories:
        cat_images = list(category_to_images[cat_id])
        random.shuffle(cat_images)
        
        # Take at least 1 image for validation, but respect the ratio
        n_val = max(1, int(len(cat_images) * val_ratio))
        
        val_images.update(cat_images[:n_val])
    
    # Add remaining images to training
    train_images = all_images - val_images
    
    # Adjust if validation set is too large
    target_val_size = int(len(all_images) * val_ratio)
    if len(val_images) > target_val_size:
        excess_val = random.sample(list(val_images), len(val_images) - target_val_size)
        val_images -= set(excess_val)
        train_images.update(excess_val)
    
    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")
    print(f"Val ratio: {len(val_images) / len(all_images):.3f}")
    
    # Verify all categories are in both splits
    train_categories = set()
    val_categories = set()
    
    for img_id in train_images:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        for ann_id in ann_ids:
            train_categories.add(coco.anns[ann_id]['category_id'])
    
    for img_id in val_images:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        for ann_id in ann_ids:
            val_categories.add(coco.anns[ann_id]['category_id'])
    
    print(f"Categories in train: {len(train_categories)}")
    print(f"Categories in val: {len(val_categories)}")
    print(f"Missing from train: {set(all_categories) - train_categories}")
    print(f"Missing from val: {set(all_categories) - val_categories}")
    
    return train_images, val_images

def create_coco_subset(coco, image_ids, output_file):
    """Create a COCO format subset with specified images"""
    # Create new COCO structure
    new_coco = {
        "info": coco.dataset.get("info", {}),
        "licenses": coco.dataset.get("licenses", []),
        "categories": coco.dataset["categories"],
        "images": [],
        "annotations": []
    }
    
    # Add images
    for img_id in image_ids:
        if img_id in coco.imgs:
            new_coco["images"].append(coco.imgs[img_id])
    
    # Add annotations for these images
    ann_id_counter = 1
    for img_id in image_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        for ann_id in ann_ids:
            ann = copy.deepcopy(coco.anns[ann_id])
            ann["id"] = ann_id_counter
            new_coco["annotations"].append(ann)
            ann_id_counter += 1
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(new_coco, f)
    
    print(f"COCO subset saved to {output_file}")
    print(f"  Images: {len(new_coco['images'])}")
    print(f"  Annotations: {len(new_coco['annotations'])}")

def create_odvg_subset(coco, image_ids, id_to_label, id_to_name, output_file):
    """Create ODVG format subset with specified images"""
    metas = []
    
    for img_id in tqdm(image_ids, desc="Creating ODVG subset"):
        if img_id not in coco.imgs:
            continue
            
        img_info = coco.imgs[img_id]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        instance_list = []
        
        for ann_id in ann_ids:
            ann = coco.anns[ann_id]
            bbox = ann['bbox']
            bbox_xyxy = coco_to_xyxy(bbox)
            
            original_category_id = ann['category_id']
            new_label = id_to_label[original_category_id]
            category_name = id_to_name[original_category_id]
            
            instance_list.append({
                "bbox": bbox_xyxy,
                "label": new_label,
                "category": category_name
            })
        
        # Create ODVG format entry
        odvg_entry = {
            "filename": img_info["file_name"],
            "height": img_info["height"],
            "width": img_info["width"],
            "detection": {
                "instances": instance_list
            }
        }
        
        metas.append(odvg_entry)
    
    # Save to JSONL file
    with jsonlines.open(output_file, mode="w") as writer:
        writer.write_all(metas)
    
    print(f"ODVG subset saved to {output_file}")
    print(f"  Images: {len(metas)}")

def coco_split_and_convert(args):
    """Main function to split COCO dataset and convert formats"""
    print(f"Loading COCO dataset from {args.input}")
    coco = COCO(args.input)
    
    # Create label mapping
    id_to_label, id_to_name = create_label_mapping(coco, start_from_zero=not args.keep_original_ids)
    
    # Save label mapping if requested
    if args.save_label_map:
        dump_label_map(id_to_label, id_to_name, args.save_label_map)
    
    # Perform stratified split
    print("\nPerforming stratified train-val split...")
    train_images, val_images = stratified_split(coco, val_ratio=args.val_ratio, seed=args.seed)
    
    # Create training set in ODVG format
    print(f"\nCreating training set in ODVG format...")
    create_odvg_subset(coco, train_images, id_to_label, id_to_name, args.train_output)
    
    # Create validation set in COCO format
    print(f"\nCreating validation set in COCO format...")
    create_coco_subset(coco, val_images, args.val_output)
    
    print(f"\nConversion complete!")
    print(f"Training (ODVG): {args.train_output}")
    print(f"Validation (COCO): {args.val_output}")
    if args.save_label_map:
        print(f"Label mapping: {args.save_label_map}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Split COCO dataset and convert to ODVG format for training", add_help=True)
    parser.add_argument("--input", '-i', required=True, type=str, 
                       help="Input COCO JSON file path")
    parser.add_argument("--train-output", required=True, type=str, 
                       help="Output training ODVG JSONL file path")
    parser.add_argument("--val-output", required=True, type=str, 
                       help="Output validation COCO JSON file path")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                       help="Validation split ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible splits (default: 42)")
    parser.add_argument("--keep-original-ids", action="store_true", default=False,
                       help="Keep original category IDs instead of starting from 0")
    parser.add_argument("--save-label-map", type=str, default=None,
                       help="Save label mapping to JSON file (optional)")
    
    args = parser.parse_args()
    coco_split_and_convert(args)