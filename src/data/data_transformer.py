import torchvision.transforms as transforms

# this data transformation is used to train our cnn from scratch
data_transform_from_scratch = transforms.Compose([transforms.Resize((224, 224)),
                                                  transforms.RandomAffine(degrees=10,
                                                                          translate=(0.15, 0.15),
                                                                          scale=(0.8, 1.2)),
                                                  transforms.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.5),
                                                  transforms.RandomGrayscale(p=0.1),
                                                  transforms.RandomHorizontalFlip(p=0.5),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

# this transformation is used on the validation and test data sets. We only resize and normalize
data_transform_bare = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
