# personal-care-action-recognition

Reconhecimento de ações relacionadas a cuidados pessoais a partir de vídeos.  
Este projeto utiliza deep learning para classificar ações em vídeos, focando na identificação de atividades de autocuidado, como escovar os dentes, aplicar maquiagem, cortar o cabelo, entre outros.

---

## 📦 Estrutura do Projeto

A seguir, uma visão geral da estrutura do projeto e da função de cada componente:

```
personal-care-action-recognition/
│
├── checkpoints/                        # Pesos dos modelos salvos durante o treinamento
│
├── data/                               # Dados utilizados no projeto
│   ├── Raw/                            # Dataset puro, não processado
│   ├── Split/                          # Versão processada do dataset
|       ├── split_info/                 # Arquivos de anotação para treino e teste
|       ├── split_info-filtered/        # Versão filtrada dos splits (apenas classes de interesse)
|       ├── UCF101/                     # Dataset original UCF101 organizado por classes
|                  
│
├── docs/                               # Documentação de planejamento do projeto
│   ├── planejamento.md             
│   └── planejamento.pdf            
│
├── notebooks/                          # Notebooks de desenvolvimento e análise
│   ├── environment_validation.ipynb    # Teste de validação do ambiente e dependências
│   ├── EDA.ipynb                       # Análise exploratória dos dados
│   ├── r3d_18_training.ipynb           # Treinamento do modelo R3D_18
│   └── video_classification.ipynb      # Testes e geração dos vídeos finais + JSON
│
├── outputs/                        # Vídeos processados e arquivos JSON gerados
│
├── src/                            # Arquivos de código principais do projeto
│   ├── data/                       # Funções de preparação dos dados
│   │   ├── data_preparation.py
│   │   └── ucf101_dataset.py
│   ├── train/                      # Funções de treinamento do modelo
│   │   ├── evaluate.py
│   │   ├── model.py
│   │   └── train.py
│   ├── utils/                      # Utilitários e funções de apoio
│   |      └── utils.py
│   |
|   └── video/                          # Processamento de vídeos
│       └── video_process.py
│
├── inference.py                    # Script para realizar inferência em novos vídeos
│
├── requirements.txt                # Dependências do projeto
└── README.md                       # Este arquivo
```

---

## 🚀 Como rodar o projeto

### 1. Clone o repositório

```bash
git clone https://github.com/GabrielGadelha3798/personal-care-action-recognition
cd personal-care-action-recognition
```

### 2. Crie o ambiente virtual e ative

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Valide o ambiente (opcional)

Abra e execute o notebook:

```
notebooks/environment_validation.ipynb
```

---

## 🧩 Funcionalidades principais

- **Análise de dados**: Notebook `notebooks/EDA.ipynb` para análise exploratória da base de dados.
- **Treinamento**: Utilize o notebook `notebooks/r3d_18_training.ipynb` para treinar o modelo utilizando a arquitetura R3D_18 pré-treinada no dataset Kinetics-400.
- **Inferência**: Execute o script `inference.py` para realizar a inferência sobre um vídeo de entrada, gerando o vídeo com as predições e os JSONs correspondentes.
- **Geração de vídeos finais**: Notebook `notebooks/video_classification.ipynb` para testes e criação dos vídeos finais processados.

---

## 📄 Planejamento

Todo o planejamento e roadmap do projeto estão documentados no arquivo:

```
docs/planejamento.pdf
```

Nele estão descritos:
- Etapas do desenvolvimento
- Próximos passos
- Decisões de arquitetura e modelo
- Desafios encontrados e soluções aplicadas

---

## 🗂️ Dados

- A pasta `data/Raw` contém o dataset original bruto.
- A pasta `data/Split` possui o dataset organizado e separado para treino e teste, de acordo com os arquivos de anotação disponíveis em `split_info/`.
- Cada classe é representada por uma pasta contendo vídeos das ações correspondentes.

---

## 💡 Observações finais

- O projeto foi desenvolvido em Python com PyTorch e foco na manipulação de vídeos.
- Os resultados gerados são salvos na pasta `outputs/`, incluindo vídeos com as predições visuais e arquivos JSON detalhando as classes e pontuações.