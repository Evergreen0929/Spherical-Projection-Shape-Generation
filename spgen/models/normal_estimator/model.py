"""Light-weight point-cloud normal estimator (3D UNet) used for Poisson surface extraction.

The network follows the encoder / decoder of Shape As Points: point features are scattered into a
32^3 grid, processed by a 3D UNet and decoded back to a unit normal per input point.
"""
import os

import torch
import torch.nn as nn

from spgen.models.normal_estimator.decoder import LocalDecoder
from spgen.models.normal_estimator.encoder import LocalPoolPointnet
from spgen.models.normal_estimator.layers import map2local

WEIGHTS_NAME = "normal_estimator.pt"


class NormalEstimator(nn.Module):
    def __init__(self, hidden_dim=192, f_maps=128, num_levels=5, grid_resolution=32, pcd_scale=1.2):
        """
        Args:
            hidden_dim: feature dimension of the point encoder / decoder.
            f_maps: number of feature maps of the first 3D UNet level.
            num_levels: number of 3D UNet levels.
            grid_resolution: resolution of the feature grid.
            pcd_scale: input points in [-pcd_scale / 2, pcd_scale / 2] are mapped to [0, 1].
        """
        super().__init__()
        self.config = dict(hidden_dim=hidden_dim, f_maps=f_maps, num_levels=num_levels,
                           grid_resolution=grid_resolution, pcd_scale=pcd_scale)
        self.pcd_scale = pcd_scale

        unet3d_kwargs = {
            "num_levels": num_levels,
            "f_maps": f_maps,
            "in_channels": hidden_dim,
            "out_channels": hidden_dim,
        }
        self.encoder = LocalPoolPointnet(
            dim=3, c_dim=hidden_dim, hidden_dim=hidden_dim,
            plane_type="grid", grid_resolution=grid_resolution,
            unet3d=True, unet3d_kwargs=unet3d_kwargs,
            map2local=map2local(1 / grid_resolution),
        )
        self.decoder_normal = LocalDecoder(dim=3, c_dim=hidden_dim, out_dim=3, sample_mode="bilinear",
                                           hidden_size=hidden_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.apply(init_weights)

    def forward(self, points):
        """
        Args:
            points: (B, N, 3) un-oriented points of a shape normalized to [-0.5, 0.5].
        Returns:
            (B, N, 3) unit normals.
        """
        # torch_scatter does not support reduced precision, the model always runs in fp32
        with torch.autocast(device_type=points.device.type, enabled=False):
            p = points.float() / self.pcd_scale + 0.5
            c = self.encoder(p)
            normals = self.decoder_normal(p.clone(), c)
            return normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save({"config": self.config, "state_dict": self.state_dict()}, os.path.join(path, WEIGHTS_NAME))

    @classmethod
    def from_pretrained(cls, path, map_location="cpu"):
        ckpt = torch.load(os.path.join(path, WEIGHTS_NAME), map_location=map_location, weights_only=True)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model.eval()
