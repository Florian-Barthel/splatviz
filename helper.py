import torch

from gaussian_splatting.scene.gaussian_model import GaussianModel


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    q: (..., 4), assumed [w, x, y, z]
    returns: (..., 3, 3)
    """
    q = torch.nn.functional.normalize(q, dim=-1)

    w, x, y, z = q.unbind(-1)

    R = torch.stack([
        1 - 2 * (y * y + z * z),
        2 * (x * y - z * w),
        2 * (x * z + y * w),

        2 * (x * y + z * w),
        1 - 2 * (x * x + z * z),
        2 * (y * z - x * w),

        2 * (x * z - y * w),
        2 * (y * z + x * w),
        1 - 2 * (x * x + y * y),
    ], dim=-1)

    return R.reshape(*q.shape[:-1], 3, 3)


@torch.no_grad()
def color_gaussians_by_shortest_axis(
    gs: GaussianModel,
    cam_params: torch.Tensor,
    view_dependent: bool = True,
):
    device = gs._xyz.device
    dtype = gs._xyz.dtype
    scales = gs._scaling
    rotations = gs._rotation

    R = quat_to_rotmat(rotations)

    # Index of smallest local scale axis.
    shortest_axis_idx = torch.argmin(scales, dim=-1)  # [N]
    normals = R[torch.arange(R.shape[0], device=device), :, shortest_axis_idx]                                           # [N, 3]
    normals = torch.nn.functional.normalize(normals, dim=-1)

    if view_dependent:
        # cam_params is 4x4 camera-to-world extrinsics
        cam_pos = cam_params[:3, 3].to(device=device, dtype=dtype)

        view_dir = cam_pos[None, :] - gs._xyz
        view_dir = torch.nn.functional.normalize(view_dir, dim=-1)

        # Flip normals toward camera
        facing = (normals * view_dir).sum(dim=-1, keepdim=True)
        normals = torch.where(facing < 0.0, -normals, normals)

    # Normal visualization: [-1, 1] -> [0, 1]
    colors = normals * 0.5 + 0.5
    colors = colors.clamp(0.0, 1.0)

    gs._features_dc = colors[:, None, :].contiguous()
    gs._features_rest = torch.zeros_like(gs._features_rest)

    return normals, colors

@torch.no_grad()
def color_gaussians_by_facing_direction(
    gs: GaussianModel,
    cam_params: torch.Tensor,
):
    device = gs._xyz.device
    dtype = gs._xyz.dtype

    scales = gs._scaling
    rotations = gs._rotation

    R = quat_to_rotmat(rotations)

    shortest_axis_idx = torch.argmin(scales, dim=-1)

    normals = R[torch.arange(R.shape[0], device=device), :, shortest_axis_idx]

    normals = torch.nn.functional.normalize(normals, dim=-1)

    # cam_params is 4x4 camera-to-world
    cam_pos = cam_params[:3, 3].to(device=device, dtype=dtype)

    view_dir = cam_pos[None, :] - gs._xyz
    view_dir = torch.nn.functional.normalize(view_dir, dim=-1)

    # Positive means normal points toward camera
    facing = (normals * view_dir).sum(dim=-1, keepdim=True)

    front_facing = facing > 0.0

    blue = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    red  = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)

    # Blue = correct / facing camera
    # Red  = back-facing / likely wrong normal
    colors = torch.where(front_facing, red, blue)

    gs._features_dc = colors[:, None, :].contiguous()
    gs._features_rest = torch.zeros_like(gs._features_rest)

    return normals, facing, colors