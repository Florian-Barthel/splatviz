import os
from compression.codecs.base_codec import BaseCodec

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2


# parameters:
# type: ["half", "float"]
# compression: ["none", "rle", "zps", "zip", "piz", "pxr24", "b4a", "b44", "dwaa", "dwab"]


class EXRCodec(BaseCodec):

    def encode_image(self, image, out_file, type="half", compression="none"):

        imwrite_flags = []

        match type:
            case "half":
                imwrite_flags.extend([cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
            case "float":
                imwrite_flags.extend([cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
            case _:
                raise NotImplementedError(f"Unknown type: {type}")

        match compression:
            case "rle":
                imwrite_flags.extend([cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_RLE])
            case "zps":
                imwrite_flags.extend([cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP])
            case "zip":
                imwrite_flags.extend([cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP])
            case "piz":
                imwrite_flags.extend([cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_PIZ])
            case "pxr24":
                imwrite_flags.extend([cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_PXR24])
            case "b4a":
                imwrite_flags.extend([cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_B44])
            case "b44":
                imwrite_flags.extend([cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_B44A])
            case "dwaa":
                imwrite_flags.extend([cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_DWAA])
            case "dwab":
                imwrite_flags.extend([cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_DWAB])
            case "none":
                pass
            case _:
                raise NotImplementedError(f"Unknown compression method: {compression}")

        cv2.imwrite(out_file, image, imwrite_flags)

    def decode_image(self, file_name):
        return cv2.imread(file_name, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)

    def file_ending(self):
        return "exr"
