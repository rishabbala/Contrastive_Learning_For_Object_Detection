class LoadVOCDataset(Dataset):


    def __init__(self, foldername, transform=None):
        self.foldername = foldername
        self.transform = transform

        self.img_dir = foldername + '/PNGImages/' 
        self.annotations = foldername + '/Annotations/'
        image_batch = []

        for img in os.listdir(foldername):
            image = torchvision.io.read_image(img)
            image_batch.append(image)

    
    def __len__(self):
        return len(os.listdir(self.foldername))

    
    def __getitem__(self, idx):
