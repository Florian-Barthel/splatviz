class Preset:
    def __init__(self, name, code, slider=[]):
        self.code = code
        self.slider = slider
        self.name = name


default = Preset(
    name="default",
    code="""gaussian._xyz = gaussian._xyz
gaussian._rotation = gaussian._rotation
gaussian._scaling = gaussian._scaling
gaussian._opacity = gaussian._opacity
gaussian._features_dc = gaussian._features_dc
gaussian._features_rest = gaussian._features_rest
self.bg_color[:] = 0
""",
)
