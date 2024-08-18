import numpy as np
from compression.codecs.base_codec import BaseCodec


class NpzCodec(BaseCodec):

    def encode_image(self, image, out_file, **kwargs):
        return np.savez_compressed(out_file, image, **kwargs)

    def decode_image(self, file_name):
        return np.load(file_name)["arr_0"]

    def file_ending(self):
        return "npz"
