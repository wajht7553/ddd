from torchvision import transforms


def get_view_transform(args):
    view_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.input_size, scale=(args.min_scale, 1.0)),
            transforms.RandomHorizontalFlip(),
            (
                transforms.ColorJitter(
                    brightness=args.cj_bright,
                    contrast=args.cj_contrast,
                    saturation=args.cj_sat,
                    hue=args.cj_hue,
                )
                if args.cj_prob > 0
                else transforms.Lambda(lambda x: x)
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=23)], p=args.gaussian_blur_prob
            ),
            transforms.RandomApply(
                [transforms.RandomSolarize(128)], p=args.solarization_prob
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return view_transform
