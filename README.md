# Gaussian Viewer
This interactive viewer allows to display 3D Gaussian Splatting objects
stored as .ply files or as compressed objects from Compact 3D Gaussian Splatting.
The viewer is fully written in python, which makes development and debugging very simple.

<img src="images/full.png">

## Install
Clone repository **recursively** in order to install glm from the diff_gaussian_rasterization package.
```
git clone "url" --recursive
```

Create conda environment:

```
conda env create -f environment.yml
conda activate gs-view
```

Launch:

`
python main.py --data_path="path/to/directory/with/gaussian/objects"
`

## Current Features

Currently, the viewer is equipped with the following features:
- Edit Widget
- Evaluate Widget
- Video Widget

### Edit Widget
The edit widget allows to modify the Gaussian object interactively. The written code
is executed just before rendering the Gaussian object. All existing objects at that 
state can be accessed and modified. An **example** looks as follows:

````py
mask = torch.linalg.norm(gaussian._scaling, dim=1) < 10
gaussian._xyz = gaussian._xyz[mask, ...]
gaussian._rotation = gaussian._rotation[mask, ...]
gaussian._scaling = gaussian._scaling[mask, ...]
gaussian._opacity = gaussian._opacity[mask, ...]
gaussian._features_dc = gaussian._features_dc[mask, ...]
gaussian._features_rest = gaussian._features_rest[mask, ...]
````
this removes all gaussian splats with a scale greater than 10.

You can also create slider variables that can be accessed in the code block with slider.varname. **Example**:

<img src="images/slider.png" style="width: 400px">


### Evaluate Widget
The evaluate widget can be used to debug the gaussian splatting object. By typing
python code, which is executed after the rendering, you can access any variable 
from the rendering context.

<img src="images/eval.png" style="width: 400px">

### Video Widget
The video widget creates a video sequence of a full rotation around the current object.
Simply define the height of the camera and the rendering resolution. While the video is
rendering, the UI is frozen. A loading screen is shown in the terminal output.

## Contribute
You are more than welcome to add further functionality to this interactive viewer!

## References
This viewer is inspired by the visualizer from Efficient Geometry-aware 3D Generative Adversarial 
Networks (EG3D):
```
@inproceedings{Chan2022,
  author = {Eric R. Chan and Connor Z. Lin and Matthew A. Chan and Koki Nagano and Boxiao Pan and Shalini De Mello and Orazio Gallo and Leonidas Guibas and Jonathan Tremblay and Sameh Khamis and Tero Karras and Gordon Wetzstein},
  title = {Efficient Geometry-aware {3D} Generative Adversarial Networks},
  booktitle = {CVPR},
  year = {2022}
}
```

3D Gaussian Splatting
```
@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```
