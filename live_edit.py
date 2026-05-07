from gaussian_splatting.scene.gaussian_model import GaussianModel
from splatviz_utils.dict_utils import EasyDict
import torch

slider: EasyDict
gs: GaussianModel


gs._xyz = gs._xyz
gs._rotation = gs._rotation
gs._scaling = gs._scaling
gs._opacity = gs._opacity
gs._features_dc = gs._features_dc
gs._features_rest = gs._features_rest
