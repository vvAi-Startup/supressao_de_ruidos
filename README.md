# Sistema de Supressão de Ruídos

Este projeto implementa um sistema de supressão de ruídos em áudio utilizando redes neurais profundas. O sistema é capaz de processar arquivos de áudio, remover ruídos indesejados e melhorar a qualidade do som.

## Características

- Interface gráfica amigável usando PyQt5
- Processamento de áudio em tempo real
- Modelo de rede neural pré-treinado para supressão de ruídos
- Suporte para diferentes formatos de áudio
- Visualização do espectro de frequência
- Métricas de qualidade de áudio (PESQ e STOI)

## Requisitos do Sistema

- Python 3.7 ou superior
- Sistema operacional Windows/Linux/MacOS
- Placa de som compatível
- Espaço em disco para o dataset e modelo

## Dependências

O projeto utiliza as seguintes bibliotecas Python:

- numpy
- torch
- torchaudio
- librosa
- matplotlib
- tqdm
- scikit-learn
- pyloudnorm
- pystoi
- PyQt5
- soundfile

## Instalação

1. Clone o repositório:
```bash
git clone [URL_DO_REPOSITÓRIO]
cd supressao_de_ruidos
```

2. Crie um ambiente virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Como Usar

1. Execute o script principal:
```bash
python app_denoiser_2.py
```

2. Na interface gráfica:
   - Clique em "Selecionar Arquivo" para escolher um arquivo de áudio
   - Use os controles para ajustar os parâmetros de processamento
   - Clique em "Processar" para iniciar a supressão de ruídos
   - Salve o resultado usando o botão "Salvar"

## Treinamento do Modelo

O modelo pode ser treinado usando os scripts:
- `treinamento_2.py`
- `treinamento_3_sem_pesq.py`

Para treinar um novo modelo:
1. Execute o script de download do dataset:
```bash
python download_dataset.py
```

2. Execute o script de treinamento desejado:
```bash
python treinamento_2.py
```

## Estrutura do Projeto

- `app_denoiser_2.py`: Interface gráfica principal
- `treinamento_2.py`: Script de treinamento com métricas PESQ
- `treinamento_3_sem_pesq.py`: Script de treinamento alternativo
- `download_dataset.py`: Script para download do dataset
- `best_denoiser_model.pth`: Modelo pré-treinado
- `requirements.txt`: Lista de dependências
- `start.sh`: Script de inicialização

## Contribuições

Contribuições são bem-vindas! Por favor, abra uma issue para discutir mudanças propostas ou envie um pull request.

## Funcionamento Detalhado

### Arquitetura do Modelo

O sistema utiliza uma rede neural profunda baseada em U-Net, uma arquitetura popular para tarefas de processamento de sinais. O modelo é composto por:

1. **Camada de Entrada**: 
   - Recebe o espectrograma do áudio ruidoso
   - Utiliza transformada de Fourier de curto prazo (STFT) para converter o sinal temporal em representação frequencial

2. **Camadas de Encoder**:
   - Reduzem a dimensionalidade do sinal
   - Extraem características relevantes
   - Utilizam convoluções e max-pooling

3. **Camadas de Decoder**:
   - Reconstroem o sinal limpo
   - Utilizam convoluções transpostas e upsampling
   - Implementam conexões residuais para preservar detalhes

4. **Camada de Saída**:
   - Gera a máscara de supressão de ruído
   - Aplica a máscara ao espectrograma original
   - Converte de volta para o domínio do tempo usando ISTFT

### Aplicação Principal (app_denoiser_2.py)

A interface gráfica foi desenvolvida usando PyQt5 e oferece as seguintes funcionalidades:

1. **Interface Principal**:
   - Visualização do espectrograma original e processado
   - Gráfico de onda do áudio
   - Controles de reprodução e processamento
   - Indicadores de métricas de qualidade (PESQ e STOI)

2. **Processamento de Áudio**:
   - Carregamento de arquivos de áudio em diferentes formatos
   - Pré-processamento do sinal (normalização, remoção de DC offset)
   - Aplicação do modelo de supressão de ruído
   - Pós-processamento (normalização de loudness)

3. **Visualização**:
   - Espectrogramas em tempo real
   - Formas de onda antes e depois do processamento
   - Gráficos de métricas de qualidade

4. **Controles**:
   - Ajuste de parâmetros de processamento
   - Controle de reprodução (play, pause, stop)
   - Navegação pelo áudio
   - Exportação do resultado

### Fluxo de Processamento

1. **Entrada do Áudio**:
   - O usuário seleciona um arquivo de áudio
   - O sistema carrega e normaliza o sinal
   - O áudio é dividido em janelas para processamento

2. **Processamento**:
   - Cada janela é convertida para o domínio da frequência
   - O modelo aplica a máscara de supressão
   - O sinal é reconstruído no domínio do tempo

3. **Pós-processamento**:
   - Normalização do volume
   - Remoção de artefatos
   - Aplicação de filtros de suavização

4. **Avaliação**:
   - Cálculo das métricas de qualidade
   - Visualização dos resultados
   - Opção de salvar o áudio processado

### Métricas de Qualidade

O sistema utiliza duas métricas principais para avaliar a qualidade do áudio:

1. **PESQ (Perceptual Evaluation of Speech Quality)**:
   - Avalia a qualidade percebida do áudio
   - Considera aspectos como clareza e naturalidade
   - Escala de 1 a 4.5 (quanto maior, melhor)

2. **STOI (Short-Time Objective Intelligibility)**:
   - Mede a inteligibilidade do áudio
   - Foca na preservação da informação do sinal
   - Valores entre 0 e 1 (quanto maior, melhor) 