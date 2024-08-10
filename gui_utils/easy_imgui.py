from imgui_bundle import imgui


def label(label_text: str, width: int = 0):
    imgui.text(label_text)
    imgui.same_line(width)


def slider(value, id, min, max, format="%.3f", log=False):
    _, value = imgui.slider_float(
        label=f"##{id}",
        v=value,
        v_min=min,
        v_max=max,
        format=format,
        flags=imgui.SliderFlags_.logarithmic.value if log else 0,
    )
    return value


def checkbox(value, id):
    _, value = imgui.checkbox(label=f"##{id}", v=value)
    return value

