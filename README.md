<div align="center">

  <h1><img src="resources/images/icon.png" width="35"> splatviz</h1>

![GitHub top language](https://img.shields.io/github/languages/top/Florian-barthel/splatviz) ![GitHub Release](https://img.shields.io/github/v/release/Florian-Barthel/splatviz) ![GitHub last commit](https://img.shields.io/github/last-commit/Florian-Barthel/splatviz) ![Static Badge](https://img.shields.io/badge/Platform-Linux-green) ![Static Badge](https://img.shields.io/badge/Platform-Windows-green)

</div>

<div align="justify"> 

This interactive viewer allows you to display and edit 3D Gaussian Splatting scenes in real time. By using the python GUI 
library (<a href="https://pyimgui.readthedocs.io/en/latest/guide/first-steps.html">pyimgui</a>) we can directly manipulate
the Gaussian python object **just before rendering**. This enables endless editing and visualization possibilities. 

Additionally, you can visualize multiple scenes
at the same time by **comparing them side by side** or in a splitscreen view or evaluate python expressions on the resulting scene.

See some example editings in this  <a href="https://www.youtube.com/watch?v=AJGSa_wnZS8&ab_channel=FlorianBarthel">
video</a>.

**NEW UPDATE:** GAN related features were moved to [https://github.com/Florian-Barthel/ganviz](https://github.com/Florian-Barthel/ganviz)

</div>

![](resources/images/teaser.gif)

## Install

Install `uv` first if it is not already available. See the official
[uv installation guide](https://docs.astral.sh/uv/getting-started/installation/), or use the standalone installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install the CUDA Toolkit version that matches the `uv` group you plan to use:

| `uv` group | CUDA Toolkit |
| --- | --- |
| `cuda118` | [CUDA Toolkit 11.8.0](https://developer.nvidia.com/cuda-11-8-0-download-archive) |
| `cuda121` | [CUDA Toolkit 12.1.1](https://developer.nvidia.com/cuda-12-1-1-download-archive) |
| `cuda124` | [CUDA Toolkit 12.4.1](https://developer.nvidia.com/cuda-12-4-1-download-archive) |
| `cuda126` | [CUDA Toolkit 12.6.3](https://developer.nvidia.com/cuda-12-6-3-download-archive) |
| `cuda128` | [CUDA Toolkit 12.8.2](https://developer.nvidia.com/cuda-12-8-2-download-archive) |
| `cuda129` | [CUDA Toolkit 12.9.1](https://developer.nvidia.com/cuda-12-9-1-download-archive) |
| `cuda130` | [CUDA Toolkit 13.0.3](https://developer.nvidia.com/cuda-13-0-3-download-archive) |

After installing `uv` and CUDA, clone and install splatviz:

```bash
git clone https://github.com/Florian-Barthel/splatviz.git
cd splatviz

uv sync --group <cuda-group>
```

Replace `<cuda-group>` with the `uv` CUDA group that matches your system:

```bash
uv sync --group cuda118
uv sync --group cuda121
uv sync --group cuda124
uv sync --group cuda126
uv sync --group cuda128
uv sync --group cuda129
uv sync --group cuda130
```


## Launch

In some cases you will have to add this variable so that opengl uses the correct version:
`export MESA_GL_VERSION_OVERRIDE=3.3`

### View 3D scenes
The default mode loads and renders standard 3DGS scenes stored in ply files or compressed yml files.

```bash
python run_main.py
```

Download example scenes [here](https://drive.google.com/drive/folders/1RNMis5ibSrl2a87ZX614ZPsHKaY6ams6?usp=sharing).

### Attach to Training

Attaches to a running training (local or remote). By default, the splatviz listens to the local host _127.0.0.1:6007_, which is used by 3DGS. Modify the host and port with `--host` and `--port`.
```bash
python run_main.py --mode=attach
```

Optional: To enable editing the scene during training, follow the steps
described [here](https://github.com/Florian-Barthel/splatviz_network).

## Widgets

<div align="justify">

### 🛠️ Edit Widget
The edit widget is the core functionality of this 3D viewer. It allows for real time editing of the gaussian
python object during runtime. The python code that you type in this text area will be executed just before rendering.
An example could look like this, where all scales are set to -8 (before activating with _exp()_) and all opacity values are set to 10 (before activating with
_sigmoid()_), while also the background is set to 1, which corresponds to white.

```python
gs._scaling = gs._scaling * 0 - 8
gs._opacity = gs._opacity * 0 + 10
```

To enable smooth editing transitions, you can create sliders (press **Add Slider**) which you can access in the
editor text by typing slider.name (eg. slider.x). An example could look as follows. Here, we create a boolean mask
that filters all gaussians that are smaller than the value stored in **slider.x**.

```python
mask = torch.linalg.norm(gs._scaling, dim=-1) < slider.x

gs._xyz = gs._xyz[mask]
gs._rotation = gs._rotation[mask]
gs._scaling = gs._scaling[mask]
gs._opacity = gs._opacity[mask]
gs._features_dc = gs._features_dc[mask]
gs._features_rest = gs._features_rest[mask]
```

Lastly, you can save and load presets of code snippets so that you don't have to type the same code again after
closing the application. Those presets are stored in a .json file (presets.json).
<br>
<img src="resources/images/edit.png" style="width: 600px;">

### 🔍 Eval Widget
The evaluate widget can be used to debug the gaussian splatting object. By typing
python code, which is executed after the rendering, you can access any variable
from the rendering context and visualize them in a histogram. Some useful variables might be:

- gaussian
- render</li>
- render_cam
- self

You can also access variables that you have defined in the editor of the Edit Widget.
<br>
<img src="resources/images/eval.png" style="width: 600px;">


### 🎥 Camera Widget
In the camera widget you can define the type and parameters of the camera. Most importantly, you can choose between the
two modes Orbit and WASD.
In **Orbit** mode, the camera is looking at a specific point in 3D space, and you control the pitch and yaw of the
camera rotating on a sphere around that point by dragging with the mouse over the rendered view. And in **WASD** mode, 
you fly through the scene using the mouse and the WASD keys similar to the camera controls in Unity.
</div>

## ⭐ Recent Features

**_Version 1.2.0_**

- 2025-09-20: Fix eval widget.
- 2025-09-20: Fix import of diff-gaussian-splatting package.
- 2025-09-20: Simplify the camera widget.
- 2025-09-20: Refactor scene loading widget to load the scenes from a file browser.
- 2025-09-18: Move GAN mode to a new repository: [https://github.com/Florian-Barthel/ganviz](https://github.com/Florian-Barthel/ganviz)
- 2025-05-26: Added GAN mode.

**_Version 1.1.0_**

- 2024-08-12: Added a new Training Widget to inspect live training stats and to pause training
- 2024-08-11: Attach to a running 3DGS training
- 2024-08-10: Refactor rendering class for easier addition of new renderer
- 2024-08-07: Better Mouse Control (translate with middle mouse button)
- 2024-08-05: Allow editing of existing sliders
- 2024-07-30: Store slider values in presets
- 2024-07-28: New Performance Widget
- 2024-07-28: Editor now highlights special variables (gs, self, slider) and gives tooltips

**_Version 1.0.0_**

- 2024-07-12: Rebuild the whole application with imgui_bundle
- 2024-07-05: Compare two or more Gaussian Splatting scenes side by side


## Citation

If you find this viewer useful, please consider citing our work:

```
@misc{barthel2024gaussian,
    title={Gaussian Splatting Decoder for 3D-aware Generative Adversarial Networks}, 
    author={Florian Barthel and Arian Beckmann and Wieland Morgenstern and Anna Hilsmann and Peter Eisert},
    year={2024},
    eprint={2404.10625},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## References

This viewer is inspired by the visualizer from Efficient Geometry-aware 3D Generative Adversarial
Networks (EG3D).

- GUI: <a href="https://pyimgui.readthedocs.io/en/latest/guide/first-steps.html">pyimgui</a> and
<a href="https://github.com/pthom/imgui_bundle">imgui_bundle</a> which are python wrappers for the c++ library
<a href="https://github.com/ocornut/imgui">ImGUI</a>
- Original code base: <a href="https://github.com/NVlabs/eg3d">EG3D</a>
- 3DGS: <a href="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/"> 3D Gaussian Splatting</a>
- Compressing 3DGS scenes: <a href="https://fraunhoferhhi.github.io/Self-Organizing-Gaussians/">Compact 3D Scene
Representation via Self-Organizing Gaussian Grids</a>
- 3DGS Rasterizer with depth and alpha: <a href="https://github.com/slothfulxtx/diff-gaussian-rasterization">Diff
rasterizer with depth and alpha</a>
