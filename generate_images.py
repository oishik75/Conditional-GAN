import argparse
import torch
import torchvision
import matplotlib.pyplot as plt
from model import Generator


def show_tensor_images(image_tensor, save_path=None):
    image_tensor = (image_tensor + 1) / 2
    image_grid = torchvision.utils.make_grid(image_tensor)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if save_path:
        plt.savefig(save_path)
    plt.show()

def generate_images(generator, args):
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    generator.to(device)

    # Create class labels
    if args.class_ == "all":
        labels = torch.arange(0, args.n_classes).repeat_interleave(args.n_images_per_class).int()
    else:
        labels = torch.tensor([int(args.class_)]).repeat(args.n_images_per_class)
    

    n_images = labels.shape[0]
    noise = torch.randn((n_images, args.z_dim, 1, 1)).to(device)

    with torch.no_grad():
        generated = generator(noise, labels).detach().cpu()

    show_tensor_images(generated, args.save_path)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--n_images_per_class", type=int, default=4)
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--z_dim", type=int, default=100)
    parser.add_argument("--image_channels", type=int, default=3)
    parser.add_argument("--features_g", type=int, default=64)
    parser.add_argument("--gen_embed_dim", type=int, default=100)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--class_", default="all")
    parser.add_argument("--no_cuda", action="store_true")

    args = parser.parse_args()

    assert args.model, "Please pass model checkpoint"

    generator = Generator(args.z_dim, args.image_channels, args.features_g, args.n_classes, args.gen_embed_dim)
    # Load model weights
    checkpoint = torch.load(args.model, weights_only=True)
    generator.load_state_dict(checkpoint['generator_state_dict'])

    generate_images(generator, args)


if __name__ == "__main__":
    main()