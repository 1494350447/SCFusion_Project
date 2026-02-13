import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class RGBT_Dataset(Dataset):
    """RGB-T dataset loader that returns paired IR, VIS and GT mask if available.

    Expected folder structure:
      root_dir/ir/*.png
      root_dir/vis/*.png
      root_dir/mask/*.png  (optional)
    Or provide a split file at root_dir/splits/{split}.txt listing filenames (without extensions)
    """

    def __init__(self, root_dir, split='train', transform=None, ext='.png'):
        self.root_dir = root_dir
        self.transform = transform
        self.ext = ext
        self.samples = []
        # Support multiple dataset layouts. Prefer VT5000 layout when detected.
        # VT5000 layout: root_dir/Train and root_dir/Test each containing RGB/, T/, GT/ and a train.csv/test.csv
        names = []
        # If user passed subfolder 'Train' or 'Test', move up to parent to detect base
        base_dir = root_dir
        base_name = os.path.basename(os.path.normpath(root_dir)).lower()
        parent = os.path.dirname(root_dir)
        if base_name in ('train', 'test') and os.path.isdir(os.path.join(parent, 'Train')):
            base_dir = parent

        is_vt5000 = os.path.isdir(os.path.join(base_dir, 'Train')) and os.path.isdir(os.path.join(base_dir, 'Test'))

        if is_vt5000:
            mode_dir = os.path.join(base_dir, 'Train' if split == 'train' else 'Test')
            csv_file = os.path.join(mode_dir, f'{split}.csv')
            if not os.path.exists(csv_file):
                # fallback: try lower-case name
                csv_file = os.path.join(mode_dir, f'{split}.CSV')

            if os.path.exists(csv_file):
                with open(csv_file, 'r') as f:
                    # skip header
                    header = f.readline()
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) == 0:
                            continue
                        names.append(parts[0].strip())
            else:
                # if no csv, enumerate RGB folder
                rgb_root = os.path.join(mode_dir, 'RGB')
                if os.path.isdir(rgb_root):
                    for root, _, files in os.walk(rgb_root):
                        for fn in files:
                            names.append(fn)

            # build lookup maps for RGB/T/GT files for faster matching
            def build_map(subfolder):
                mapping = {}
                folder = os.path.join(mode_dir, subfolder)
                if not os.path.isdir(folder):
                    return mapping
                for r, _, files in os.walk(folder):
                    for fn in files:
                        key = os.path.splitext(fn)[0]
                        mapping[key] = os.path.join(r, fn)
                return mapping

            rgb_map = build_map('RGB')
            t_map = build_map('T')
            gt_map = build_map('GT')

            for name in names:
                key = os.path.splitext(name)[0]
                vis_path = rgb_map.get(key)
                ir_path = t_map.get(key)
                mask_path = gt_map.get(key)
                if ir_path and vis_path:
                    self.samples.append((ir_path, vis_path, mask_path))
        else:
            ir_dir = os.path.join(root_dir, 'ir')
            vis_dir = os.path.join(root_dir, 'vis')
            mask_dir = os.path.join(root_dir, 'mask')

            split_file = os.path.join(root_dir, 'splits', f'{split}.txt')
            if os.path.exists(split_file):
                with open(split_file, 'r') as f:
                    for line in f:
                        names.append(line.strip())
            else:
                if os.path.isdir(ir_dir):
                    names = [os.path.splitext(n)[0] for n in sorted(os.listdir(ir_dir))]

            for name in names:
                ir_path = os.path.join(ir_dir, name + self.ext)
                vis_path = os.path.join(vis_dir, name + self.ext)
                mask_path = os.path.join(mask_dir, name + self.ext)
                if os.path.exists(ir_path) and os.path.exists(vis_path):
                    if os.path.exists(mask_path):
                        self.samples.append((ir_path, vis_path, mask_path))
                    else:
                        self.samples.append((ir_path, vis_path, None))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ir_path, vis_path, mask_path = self.samples[idx]
        ir = Image.open(ir_path).convert('L')
        vis = Image.open(vis_path).convert('RGB')
        has_mask = mask_path is not None
        mask = Image.open(mask_path).convert('L') if has_mask else None
        if self.transform:
            res = self.transform(ir, vis, mask) if mask is not None else self.transform(ir, vis)
            if mask is not None:
                ir_t, vis_t, mask_t = res
                return {'ir': ir_t, 'vis': vis_t, 'mask': mask_t, 'has_mask': torch.tensor(True, dtype=torch.bool)}
            else:
                ir_t, vis_t = res
                mask_t = ir_t.new_zeros((1, ir_t.shape[1], ir_t.shape[2]))
                return {'ir': ir_t, 'vis': vis_t, 'mask': mask_t, 'has_mask': torch.tensor(False, dtype=torch.bool)}
        return {'ir': ir, 'vis': vis, 'mask': mask, 'has_mask': has_mask}

