import os
import shutil
import random

def move_videos_to_class_folders(src_dir, dest_root):
    """
    Move vídeos para pastas de classes baseadas no nome do arquivo.
    
    Parâmetros:
    src_dir (str): Diretório de origem onde os vídeos estão localizados.
    dest_root (str): Diretório de destino onde as pastas de classes serão criadas.
    
    """

    # Verifica se o diretório de origem existe
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"O diretório de origem '{src_dir}' não existe.")
    
    # Verifica se o diretório de destino existe, se não, cria
    if not os.path.exists(dest_root):
        os.makedirs(dest_root)
    
    # Cria estrutura de pastas
    for filename in os.listdir(src_dir):
        if filename.endswith(".avi"):
            # Extrai nome da classe do filename (ex: "ApplyEyeMakeup" de "v_ApplyEyeMakeup_g01_c01.avi")
            class_name = filename.split("_")[1]
            
            # Cria pasta da classe
            class_dir = os.path.join(dest_root, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Move o vídeo
            src_path = os.path.join(src_dir, filename)
            dest_path = os.path.join(class_dir, filename)
            shutil.move(src_path, dest_path)

def split_data_generate_annotations(dataset_root, output_dir, class_to_label, train_ratio=0.8):
    """
    Divide os vídeos em conjuntos de treino e teste, mantendo a estratificação por grupo.

    Parâmetros:
    dataset_root (str): Diretório raiz do dataset.
    output_dir (str): Diretório onde os arquivos de treino e teste serão salvos.
    train_ratio (float): Proporção de vídeos para o conjunto de treino (0.0 a 1.0).

    """

    # Criar diretório de saída
    os.makedirs(output_dir, exist_ok=True)

    # Listar todos os vídeos e manter controle dos grupos
    videos = []
    for class_name in class_to_label:
        class_dir = os.path.join(dataset_root, class_name)
        
        # Extrair grupo de cada vídeo (ex: 'g01' de 'v_ApplyEyeMakeup_g01_c01.avi')
        for video in os.listdir(class_dir):
            if video.endswith(".avi"):
                group = video.split("_")[3][1:]  # Extrai número do grupo (ex: '01')
                videos.append({
                    "path": f"{class_name}/{video}",
                    "label": class_to_label[class_name],
                    "group": group
                })

    # Estratificar por grupo para prevenir vazamento
    groups = list(set(v["group"] for v in videos))
    random.shuffle(groups)
    split_idx = int(len(groups) * train_ratio)
    train_groups = set(groups[:split_idx])
    test_groups = set(groups[split_idx:])

    # Escrever arquivos
    with open(os.path.join(output_dir, "trainlist.txt"), "w") as f_train, \
        open(os.path.join(output_dir, "testlist.txt"), "w") as f_test:

        for video in videos:
            line = f"{video['path']} {video['label']}\n"
            
            if video["group"] in train_groups:
                f_train.write(line)
            else:
                f_test.write(line)


