from os import path
from c3dgs.scene import GaussianModel
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser("npz2ply")
    parser.add_argument("npz_file", type=str)
    parser.add_argument("--ply_file", type=str, default=None, required=False)
    args = parser.parse_args()

    if args.ply_file is None:
        file_path = path.splitext(args.npz_file)[0]
        args.ply_file = f"{file_path}.ply"

    gaussians = GaussianModel(3)
    print(f"loading '{args.npz_file}'")
    gaussians.load_npz(args.npz_file)
    print(f"saving to '{args.ply_file}'")
    gaussians.save_ply(args.ply_file)
    print("done")