import torch
from torchvision.datasets import UCF101
from torchvision.models.video import R3D_18_Weights

# O dataset retorne video, audio, label
# Audio não é necessário para o modelo R3D_18, então vamos ignorá-lo
def collate_fn(batch):
    inputs = [item[0] for item in batch]
    labels = [item[2] for item in batch]
    return torch.stack(inputs), torch.tensor(labels)

def load_ucf101_dataset(data_path, annot_path, frames_per_clip, step_between_clips, batch_size=35):
    """
    Carrega o dataset UCF101 com as transformações necessárias para o modelo R3D_18.
    
    Parâmetros:
    data_path (str): Caminho para o diretório do dataset.
    annot_path (str): Caminho para o arquivo de anotações.
    frames_per_clip (int): Número de frames por clipe.
    step_between_clips (int): Passo entre clipes.
    train_dataset (bool): Se True, carrega o conjunto de treino. Se False, carrega o conjunto de teste.

    Retorna:
    DataLoader: DataLoader para o dataset UCF101.
    """

    # Carregar pesos e transformações
    weights = R3D_18_Weights.DEFAULT
    preprocess = weights.transforms()

    train_dataset = UCF101(
        root=data_path,
        annotation_path=annot_path,
        frames_per_clip=frames_per_clip,
        step_between_clips=step_between_clips,
        output_format="TCHW",
        transform=preprocess,
        fold=1, # UCF101 original tem 3 folds, por usar uma versão customizada, usamos apenas o fold 1
        train=True
    )

    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0, 
        collate_fn=collate_fn
    )

    test_dataset = UCF101(
        root=data_path,
        annotation_path=annot_path,
        frames_per_clip=frames_per_clip,
        step_between_clips=step_between_clips,
        output_format="TCHW",
        transform=preprocess,
        fold=1,
        train=False
    )

    # DataLoader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0, 
        collate_fn=collate_fn
    )

    return train_loader, test_loader