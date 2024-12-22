from torch.utils.data import DataLoader

def get_loaders(args, dataset, train_mode = True):
    if train_mode:
        dataloader = DataLoader(
                                    dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True
                                )
        return dataloader
    dataloader = DataLoader(
                                    dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False
                                )
    return dataloader