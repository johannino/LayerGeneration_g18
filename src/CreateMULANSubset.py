import os
from collections import defaultdict
import shutil
import argparse
import struct
from PIL import Image

# Path to your dataset
source_folder = '/work3/s243345/adlcv/project/mulan_dataset_final/mulan_dataset_final'

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process MULAN dataset.")
parser.add_argument(
    "--option",
    choices=["count_layers", "create_subset", "clean", "create_last_layer", "resize"],
    required=True,
    help="Choose the operation to perform: 'count_layers' or 'create_subset'."
)
args = parser.parse_args()
option = args.option

if option == "count_layers":
    # Group files by base ID
    layer_groups = defaultdict(set)
    for filename in os.listdir(source_folder):
        if filename.endswith('.png') and '-layer_' in filename:
            base_id, layer_part = filename.split('-layer_')
            layer_index = int(layer_part.replace('.png', ''))
            layer_groups[base_id].add(layer_index)

    # Count sets based on number of layers
    count_1_layers = 0
    count_2_layers = 0
    count_3_layers = 0
    count_4_layers = 0
    count_5_layers = 0
    count_6_layers = 0

    for layers in layer_groups.values():
        if layers == {0, 1, 2, 3}:
            count_4_layers += 1
        elif layers == {0, 1, 2, 3, 4}:
            count_5_layers += 1
        elif layers == {0, 1, 2, 3, 4, 5}:
            count_6_layers += 1
        elif layers == {0, 1, 2}:
            count_3_layers += 1
        elif layers == {0, 1}:
            count_2_layers += 1
        elif layers == {0}:
            count_1_layers += 1

    print(f"1-layer sets: {count_1_layers}")
    print(f"2-layer sets: {count_2_layers}")
    print(f"3-layer sets: {count_3_layers}")
    print(f"4-layer sets: {count_4_layers}")
    print(f"5-layer sets: {count_5_layers}")
    print(f"6-layer sets: {count_6_layers}")

elif option == "create_subset":
    from collections import defaultdict
    from PIL import Image
    import os, shutil

    output_root   = '/work3/s243345/adlcv/project/MULAN_data'
    os.makedirs(output_root, exist_ok=True)

    # ensure layer_0 … layer_5 directories exist
    # 1) Make target layer dirs
    for i in range(6):
        os.makedirs(os.path.join(output_root, f'layer_{i}'), exist_ok=True)

    # 2) Group all files in the flat folder by base ID
    groups = defaultdict(list)
    for fn in os.listdir(source_folder):
        if not fn.endswith('.png') or '-layer_' not in fn:
            continue
        base, part = fn.rsplit('-layer_', 1)
        try:
            idx = int(part.split('.png')[0])
        except ValueError:
            continue
        groups[base].append((idx, fn))

    # 3) Process each group
    for base, items in groups.items():
        # need at least the background
        if not any(idx == 0 for idx, _ in items):
            continue

        # map layer index -> filename
        layer_map = {idx: fn for idx, fn in items}
        comp_img = None

        # build out layers 0..5
        for idx in range(6):
            # if you have a real layer, composite it
            if idx in layer_map:
                layer = Image.open(os.path.join(source_folder, layer_map[idx])).convert("RGBA")
                comp_img = layer if idx == 0 else Image.alpha_composite(comp_img, layer)
            # otherwise, just reuse whatever comp_img was last

            # save to the corresponding folder
            dst_dir = os.path.join(output_root, f'layer_{idx}')
            dst_path = os.path.join(dst_dir, f"{base}-layer_{idx}.png")
            comp_img.save(dst_path)

elif option == "create_last_layer":
    output_root = '/work3/s243345/adlcv/project/MULAN_data_final'

    # 1) make sure the single target folder exists
    final_dir = os.path.join(output_root, 'layer_5')
    os.makedirs(final_dir, exist_ok=True)

    # 2) group your flat files by base ID
    groups = defaultdict(list)
    for fn in os.listdir(source_folder):
        if not fn.endswith('.png') or '-layer_' not in fn:
            continue
        base, part = fn.rsplit('-layer_', 1)
        try:
            idx = int(part.split('.png')[0])
        except ValueError:
            continue
        groups[base].append((idx, fn))

    # 3) for each ID, build up through its highest layer and save only that final image
    for base, items in groups.items():
        # must at least have layer_0
        if not any(idx == 0 for idx, _ in items):
            continue

        # quick lookup
        layer_map = {idx: fn for idx, fn in items}
        max_idx = max(layer_map.keys())      # e.g. 2 if you only have up to layer_2

        comp_img = None
        for idx in range(max_idx + 1):
            fn = layer_map.get(idx)
            if fn:
                layer = Image.open(os.path.join(source_folder, fn)).convert("RGBA")
                comp_img = layer if idx == 0 else Image.alpha_composite(comp_img, layer)
            # if idx not in layer_map, we just reuse comp_img from the last existing layer

        # 4) save that final composite into layer_5/
        dst_path = os.path.join(final_dir, f"{base}-layer_5.png")
        comp_img.save(dst_path)

elif option == "clean":
    output_root   = '/work3/s243345/adlcv/project/MULAN_data'
    groups = defaultdict(set)

    # Build map: base_id -> set of layers present
    for d in os.listdir(output_root):
        if not d.startswith('layer_'): continue
        idx = int(d.split('_')[1])
        for fn in os.listdir(os.path.join(output_root, d)):
            if fn.endswith('.png') and '-layer_' in fn:
                base = fn.split('-layer_')[0]
                groups[base].add(idx)

    # Delete any incomplete sets
    for base, layers in groups.items():
        if layers != set(range(6)):
            print(f"Removing incomplete set {base} (found layers {sorted(layers)})")
            for i in layers:
                try:
                    os.remove(os.path.join(output_root, f'layer_{i}', f'{base}-layer_{i}.png'))
                except FileNotFoundError:
                    pass

elif option == "resize":
    src_dir = "/work3/s243345/adlcv/project/MULAN_data/layer_5"
    dst_dir = "/work3/s243345/adlcv/project/MULAN_data/layer_5_resized"
    os.makedirs(dst_dir, exist_ok=True)

    for fn in os.listdir(src_dir):
        if fn.endswith(".png"):
            src_path = os.path.join(src_dir, fn)
            dst_path = os.path.join(dst_dir, fn)

            try:
                img = Image.open(src_path).convert("RGB")
                img = img.resize((256, 256), Image.LANCZOS)
                img.save(dst_path)
            except Exception as e:
                print(f"⚠️ Failed to process {fn}: {e}")
