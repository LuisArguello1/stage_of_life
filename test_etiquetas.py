import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.breeds = []
        self.stages = []

        # Recorremos carpetas
        for breed in os.listdir(root_dir):
            breed_path = os.path.join(root_dir, breed)
            if os.path.isdir(breed_path):
                if breed not in self.breeds:
                    self.breeds.append(breed)

                for stage in os.listdir(breed_path):
                    stage_path = os.path.join(breed_path, stage)
                    if os.path.isdir(stage_path):
                        if stage not in self.stages:
                            self.stages.append(stage)

                        for img_name in os.listdir(stage_path):
                            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                                img_path = os.path.join(stage_path, img_name)
                                self.samples.append((img_path, breed, stage))

        # Mapear etiquetas a índices
        self.breed_to_idx = {breed: i for i, breed in enumerate(sorted(self.breeds))}
        self.stage_to_idx = {stage: i for i, stage in enumerate(sorted(self.stages))}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, breed, stage = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        breed_label = self.breed_to_idx[breed]
        stage_label = self.stage_to_idx[stage]

        return image, breed_label, stage_label


# --------------------------
# USO DEL DATASET
# --------------------------

DATA_DIR = "dataset/train"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = DogDataset(DATA_DIR, transform=transform)

print("Razas detectadas:", dataset.breeds)
print("Etapas detectadas:", dataset.stages)

img, breed_label, stage_label = dataset[0]
print("Ejemplo -> Raza:", breed_label, " Etapa:", stage_label)
