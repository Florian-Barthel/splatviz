import torch

from splatviz_utils.cam_utils import get_default_intrinsics, get_default_extrinsics


class LatentMap:
    def __init__(self, cols_rows=10, device="cuda"):
        self.device = device
        self.cols_rows = cols_rows
        self.z_map = torch.randn([1, 512, cols_rows, cols_rows], device=device, dtype=torch.float)
        self.w_map = None


    def get_latent(self, latent_x, latent_y, latent_space):
        latent_x = torch.tensor(latent_x, device=self.device, dtype=torch.float)
        latent_y = torch.tensor(latent_y, device=self.device, dtype=torch.float)
        position = torch.stack([latent_x, latent_y]).reshape(1, 1, 1, 2)
        if latent_space == "Z":
            z = torch.nn.functional.grid_sample(self.z_map, position, padding_mode="reflection")
            return z.reshape(1, 512)
        elif latent_space == "W":
            if self.w_map is None:
                raise AssertionError("call load_w_map(mapping_network) first)")
            w = torch.nn.functional.grid_sample(self.w_map, position, padding_mode="reflection")
            return w.reshape(1, 512)
        else:
            raise NotImplementedError


    def load_w_map(self, mapping_network, truncation_psi):
        intrinsics = get_default_intrinsics().to(self.device)
        extrinsics = get_default_extrinsics().to(self.device)
        mapping_camera_params = torch.concat([extrinsics.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        reshaped_z_map = self.z_map.permute(0, 2, 3, 1).reshape(-1, 512)
        mapping_camera_params = mapping_camera_params.repeat([reshaped_z_map.shape[0], 1])

        self.w_map = mapping_network(reshaped_z_map, mapping_camera_params, truncation_psi=truncation_psi)[:, 0, :]
        self.w_map = self.w_map.reshape(1, self.cols_rows, self.cols_rows, 512).permute(0, 3, 1, 2)