import os
from PIL import Image
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
        ir_dir = os.path.join(root_dir, 'ir')
        vis_dir = os.path.join(root_dir, 'vis')
        mask_dir = os.path.join(root_dir, 'mask')

        split_file = os.path.join(root_dir, 'splits', f'{split}.txt')
        names = []
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
                return {'ir': ir_t, 'vis': vis_t, 'mask': mask_t, 'has_mask': True}
            else:
                ir_t, vis_t = res
                mask_t = ir_t.new_zeros((1, ir_t.shape[1], ir_t.shape[2]))
                return {'ir': ir_t, 'vis': vis_t, 'mask': mask_t, 'has_mask': False}
        return {'ir': ir, 'vis': vis, 'mask': mask, 'has_mask': has_mask}

