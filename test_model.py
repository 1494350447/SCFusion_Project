import torch

from models.scfusion_net import SCFusion


def main():
    model = SCFusion(in_ch_ir=1, in_ch_vis=3, base_ch=32, deep_supervision=True)
    model.eval()

    ir = torch.randn(2, 1, 256, 256)
    vis = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        outputs = model(ir, vis)

    if not isinstance(outputs, (tuple, list)):
        raise RuntimeError('deep_supervision=True should return tuple/list outputs')

    for idx, out in enumerate(outputs):
        print(f'output[{idx}] shape = {tuple(out.shape)}')


if __name__ == '__main__':
    main()

