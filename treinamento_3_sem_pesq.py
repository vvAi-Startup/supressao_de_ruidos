import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pyloudnorm as pyln
import torch.nn.functional as F
from pystoi import stoi

# Configurações gerais
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 128
SEGMENT_LENGTH = SAMPLE_RATE * 2  # Segmentos de 2 segundos
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")

# Classe para o conjunto de dados
class AudioDenoisingDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, segment_length=SEGMENT_LENGTH, sample_rate=SAMPLE_RATE):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        
        # Obter lista de arquivos correspondentes
        self.clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.wav')])
        self.noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.wav')])
        
        # Verificar correspondência dos arquivos
        assert len(self.clean_files) == len(self.noisy_files), "Número desigual de arquivos limpos e ruidosos"
        for cf, nf in zip(self.clean_files, self.noisy_files):
            assert os.path.splitext(cf)[0] == os.path.splitext(nf)[0], f"Arquivos não correspondentes: {cf} e {nf}"
    
    def __len__(self):
        return len(self.clean_files)
    
    def __getitem__(self, idx):
        clean_path = os.path.join(self.clean_dir, self.clean_files[idx])
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])
        
        # Carregar arquivos de áudio
        clean_audio, sr = torchaudio.load(clean_path)
        noisy_audio, _ = torchaudio.load(noisy_path)
        
        # Converter para mono se estiver em estéreo
        if clean_audio.shape[0] > 1:
            clean_audio = torch.mean(clean_audio, dim=0, keepdim=True)
        if noisy_audio.shape[0] > 1:
            noisy_audio = torch.mean(noisy_audio, dim=0, keepdim=True)
        
        # Reamostrar se necessário
        if sr != self.sample_rate:
            clean_audio = torchaudio.transforms.Resample(sr, self.sample_rate)(clean_audio)
            noisy_audio = torchaudio.transforms.Resample(sr, self.sample_rate)(noisy_audio)
        
        # Normalizar amplitude para [-1, 1]
        clean_audio = clean_audio / (torch.max(torch.abs(clean_audio)) + 1e-8)
        noisy_audio = noisy_audio / (torch.max(torch.abs(noisy_audio)) + 1e-8)
        
        # Garantir que ambos os áudios tenham o mesmo comprimento
        min_len = min(clean_audio.shape[1], noisy_audio.shape[1])
        clean_audio = clean_audio[:, :min_len]
        noisy_audio = noisy_audio[:, :min_len]
        
        # Selecionar segmento aleatório se o áudio for maior que segment_length
        if min_len > self.segment_length:
            start = np.random.randint(0, min_len - self.segment_length)
            clean_audio = clean_audio[:, start:start+self.segment_length]
            noisy_audio = noisy_audio[:, start:start+self.segment_length]
        else:
            # Preencher com zeros se o áudio for menor que segment_length
            clean_pad = torch.zeros(1, self.segment_length)
            noisy_pad = torch.zeros(1, self.segment_length)
            clean_pad[:, :min_len] = clean_audio
            noisy_pad[:, :min_len] = noisy_audio
            clean_audio = clean_pad
            noisy_audio = noisy_pad
        
        # Calcular espectrogramas
        clean_spec = self._compute_stft(clean_audio.squeeze(0))
        noisy_spec = self._compute_stft(noisy_audio.squeeze(0))
        
        return {
            'clean_audio': clean_audio,
            'noisy_audio': noisy_audio,
            'clean_spec': clean_spec,
            'noisy_spec': noisy_spec
        }
    
    def _compute_stft(self, audio):
        # Calcular STFT
        stft = torch.stft(
            audio, 
            n_fft=N_FFT, 
            hop_length=HOP_LENGTH, 
            window=torch.hann_window(N_FFT), 
            return_complex=True
        )
        
        # Converter para magnitude e fase
        mag = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Aplicar escala logarítmica à magnitude (log1p para evitar log(0))
        log_mag = torch.log1p(mag)
        
        # Retornar magnitude e fase como tensores separados
        return {'magnitude': log_mag, 'phase': phase}


# Modelo U-Net para supressão de ruído
class UNetDenoiser(nn.Module):
    def __init__(self):
        super(UNetDenoiser, self).__init__()
        
        # Encoder (Downsampling)
        self.enc1 = self._encoder_block(1, 16)
        self.enc2 = self._encoder_block(16, 32)
        self.enc3 = self._encoder_block(32, 64)
        self.enc4 = self._encoder_block(64, 128)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        
        # Decoder (Upsampling)
        self.dec4 = self._decoder_block(256 + 128, 128)
        self.dec3 = self._decoder_block(128 + 64, 64)
        self.dec2 = self._decoder_block(64 + 32, 32)
        self.dec1 = self._decoder_block(32 + 16, 16)
        
        # Camada final para gerar a máscara
        self.final = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()  # Sigmoid para garantir que os valores da máscara estejam entre 0 e 1
        )
        
    def _encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)
        )
    
    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        # Dimensões esperadas: (batch_size, 1, freq_bins, time_frames)
        
        # Encoder
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        enc4_out = self.enc4(enc3_out)
        
        # Bottleneck
        bottleneck_out = self.bottleneck(enc4_out)
        
        # Decoder com skip connections
        # Adicionar verificação e redimensionamento para garantir dimensões compatíveis
        dec4_out = self.dec4(torch.cat([bottleneck_out, self._crop_tensor(enc4_out, bottleneck_out)], dim=1))
        dec3_out = self.dec3(torch.cat([dec4_out, self._crop_tensor(enc3_out, dec4_out)], dim=1))
        dec2_out = self.dec2(torch.cat([dec3_out, self._crop_tensor(enc2_out, dec3_out)], dim=1))
        dec1_out = self.dec1(torch.cat([dec2_out, self._crop_tensor(enc1_out, dec2_out)], dim=1))
        
        # Camada final para gerar a máscara
        mask = self.final(dec1_out)
        
        # Garantir que a máscara tenha o mesmo tamanho da entrada
        if mask.size() != x.size():
            mask = F.interpolate(mask, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        
        return mask
    
    def _crop_tensor(self, source, target):
        # Recorta o tensor fonte para corresponder às dimensões do tensor alvo
        _, _, h_source, w_source = source.size()
        _, _, h_target, w_target = target.size()
        
        h_diff = (h_source - h_target) // 2
        w_diff = (w_source - w_target) // 2
        
        if h_diff < 0 or w_diff < 0:
            # Se o alvo for maior que a fonte, redimensione a fonte
            return F.interpolate(source, size=(h_target, w_target), mode='bilinear', align_corners=False)
        
        # Caso contrário, recorte a fonte
        return source[:, :, h_diff:h_diff+h_target, w_diff:w_diff+w_target]


# Função de perda combinada
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # Peso para MSE
        self.beta = beta    # Peso para L1
        self.gamma = gamma  # Peso para perda espectral
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, predicted_spec, clean_spec, predicted_audio=None, clean_audio=None):
        # MSE na magnitude do espectrograma
        mse = self.mse_loss(predicted_spec, clean_spec)
        
        # L1 na magnitude do espectrograma
        l1 = self.l1_loss(predicted_spec, clean_spec)
        
        # Perda espectral (simplificada)
        spec_loss = torch.mean(torch.abs(torch.abs(torch.fft.rfft(predicted_audio.squeeze(1))) - 
                                       torch.abs(torch.fft.rfft(clean_audio.squeeze(1)))))
        
        # Combinação ponderada
        total_loss = self.alpha * mse + self.beta * l1 + self.gamma * spec_loss
        
        return total_loss


# Função para calcular métricas de avaliação
def calculate_metrics(clean_audio, denoised_audio, sample_rate=SAMPLE_RATE):
    clean_audio = clean_audio.squeeze().cpu().numpy()
    denoised_audio = denoised_audio.squeeze().cpu().numpy()
    
    # Normalizar para evitar erros nas métricas
    if np.max(np.abs(clean_audio)) > 0:
        clean_audio = clean_audio / np.max(np.abs(clean_audio))
    if np.max(np.abs(denoised_audio)) > 0:
        denoised_audio = denoised_audio / np.max(np.abs(denoised_audio))
    
    # SNR (Signal-to-Noise Ratio)
    def calculate_snr(clean, denoised):
        noise = clean - denoised
        signal_power = np.sum(clean ** 2)
        noise_power = np.sum(noise ** 2)
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 100  # Valor arbitrário alto para casos de ruído zero
        return snr
    
    snr_value = calculate_snr(clean_audio, denoised_audio)
    
    # STOI (Short-Time Objective Intelligibility)
    try:
        stoi_value = stoi(clean_audio, denoised_audio, sample_rate, extended=False)
    except:
        stoi_value = 0
    
    return {'SNR': snr_value, 'STOI': stoi_value}


# Reconstrução do áudio a partir da máscara e do espectrograma
def reconstruct_audio(noisy_spec, mask, hop_length=HOP_LENGTH):
    # Aplicar a máscara ao espectrograma ruidoso
    magnitude = torch.exp(noisy_spec['magnitude']) - 1  # Reverter log1p
    enhanced_magnitude = magnitude * mask.squeeze(1)
    
    # Recombinar magnitude e fase para voltar ao domínio complexo
    enhanced_stft = enhanced_magnitude * torch.exp(1j * noisy_spec['phase'])
    
    # Reconstruir o sinal de áudio usando a ISTFT
    audio = torch.istft(
        enhanced_stft, 
        n_fft=N_FFT, 
        hop_length=hop_length, 
        window=torch.hann_window(N_FFT, device=enhanced_stft.device), 
        length=SEGMENT_LENGTH
    )
    
    return audio


# Função de treinamento
def train(model, train_loader, val_loader, num_epochs, criterion, optimizer, device):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        # Loop de treinamento
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            noisy_spec = batch['noisy_spec']['magnitude'].unsqueeze(1).to(device)
            clean_spec = batch['clean_spec']['magnitude'].unsqueeze(1).to(device)
            noisy_audio = batch['noisy_audio'].to(device)
            clean_audio = batch['clean_audio'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            mask = model(noisy_spec)
            
            # Aplicar máscara ao espectrograma ruidoso
            predicted_spec = noisy_spec * mask
            
            # Reconstruir áudio para a função de perda
            predicted_audio = reconstruct_audio(
                {'magnitude': noisy_spec.squeeze(1), 'phase': batch['noisy_spec']['phase'].to(device)},
                mask,
                HOP_LENGTH
            ).unsqueeze(1)
            
            # Calcular perda
            loss = criterion(predicted_spec, clean_spec, predicted_audio, clean_audio)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calcular perda média de treinamento
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validação
        model.eval()
        val_loss = 0
        all_metrics = {'SNR': 0, 'STOI': 0}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                noisy_spec = batch['noisy_spec']['magnitude'].unsqueeze(1).to(device)
                clean_spec = batch['clean_spec']['magnitude'].unsqueeze(1).to(device)
                noisy_audio = batch['noisy_audio'].to(device)
                clean_audio = batch['clean_audio'].to(device)
                
                # Forward pass
                mask = model(noisy_spec)
                
                # Aplicar máscara ao espectrograma ruidoso
                predicted_spec = noisy_spec * mask
                
                # Reconstruir áudio para a função de perda e métricas
                predicted_audio = reconstruct_audio(
                    {'magnitude': noisy_spec.squeeze(1), 'phase': batch['noisy_spec']['phase'].to(device)},
                    mask,
                    HOP_LENGTH
                ).unsqueeze(1)
                
                # Calcular perda
                loss = criterion(predicted_spec, clean_spec, predicted_audio, clean_audio)
                val_loss += loss.item()
                
                # Calcular métricas para a primeira amostra do lote
                metrics = calculate_metrics(clean_audio[0], predicted_audio[0])
                for key in all_metrics:
                    all_metrics[key] += metrics[key]
        
        # Calcular perda média de validação e métricas
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        for key in all_metrics:
            all_metrics[key] /= len(val_loader)
        
        # Imprimir resultados
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Metrics: SNR={all_metrics['SNR']:.2f} dB, STOI={all_metrics['STOI']:.4f}")
        
        # Atualizar scheduler
        scheduler.step(val_loss)
        
        # Salvar o melhor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_denoiser_model.pth')
            print("Novo melhor modelo salvo!")
    
    # Plotar curvas de perda
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Treinamento')
    plt.plot(val_losses, label='Validação')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.title('Curvas de Perda')
    plt.legend()
    plt.savefig('loss_curves.png')
    plt.close()
    
    return train_losses, val_losses


# Função principal
def main():
    # Diretórios de dados
    clean_dir = 'dados/voz_limpa'
    noisy_dir = 'dados/ruidos'
    
    # Verificar se os diretórios existem
    if not os.path.exists(clean_dir) or not os.path.exists(noisy_dir):
        print(f"Erro: Diretórios de dados não encontrados. Certifique-se de que '{clean_dir}' e '{noisy_dir}' existem.")
        return
    
    # Obter lista de arquivos
    clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.wav')])
    
    if len(clean_files) == 0:
        print("Erro: Nenhum arquivo de áudio encontrado nos diretórios.")
        return
    
    # Dividir dados em conjuntos de treinamento, validação e teste
    train_files, test_files = train_test_split(clean_files, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=42)
    
    print(f"Arquivos para treinamento: {len(train_files)}")
    print(f"Arquivos para validação: {len(val_files)}")
    print(f"Arquivos para teste: {len(test_files)}")
    
    # Criar diretórios temporários para divisão dos dados
    os.makedirs('temp/train/clean', exist_ok=True)
    os.makedirs('temp/train/noisy', exist_ok=True)
    os.makedirs('temp/val/clean', exist_ok=True)
    os.makedirs('temp/val/noisy', exist_ok=True)
    os.makedirs('temp/test/clean', exist_ok=True)
    os.makedirs('temp/test/noisy', exist_ok=True)
    
    # Copiar arquivos para diretórios temporários
    import shutil
    for file_set, target_dir in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
        for f in file_set:
            shutil.copy(os.path.join(clean_dir, f), os.path.join(f'temp/{target_dir}/clean', f))
            shutil.copy(os.path.join(noisy_dir, f), os.path.join(f'temp/{target_dir}/noisy', f))
    
    # Criar conjuntos de dados
    train_dataset = AudioDenoisingDataset(f'temp/train/clean', f'temp/train/noisy')
    val_dataset = AudioDenoisingDataset(f'temp/val/clean', f'temp/val/noisy')
    test_dataset = AudioDenoisingDataset(f'temp/test/clean', f'temp/test/noisy')
    
    # Criar data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Inicializar modelo, função de perda e otimizador
    model = UNetDenoiser().to(DEVICE)
    criterion = CombinedLoss(alpha=0.7, beta=0.2, gamma=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Resumo do modelo
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total de parâmetros do modelo: {total_params:,}")
    
    # Treinar modelo
    train_losses, val_losses = train(model, train_loader, val_loader, NUM_EPOCHS, criterion, optimizer, DEVICE)
    
    # Carregar o melhor modelo
    model.load_state_dict(torch.load('best_denoiser_model.pth'))
    
    # Avaliar no conjunto de teste
    model.eval()
    test_metrics = {'SNR': 0, 'STOI': 0}
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Avaliando no conjunto de teste"):
            noisy_spec = batch['noisy_spec']['magnitude'].unsqueeze(1).to(DEVICE)
            clean_audio = batch['clean_audio'].to(DEVICE)
            
            # Forward pass
            mask = model(noisy_spec)
            
            # Reconstruir áudio
            predicted_audio = reconstruct_audio(
                {'magnitude': noisy_spec.squeeze(1), 'phase': batch['noisy_spec']['phase'].to(DEVICE)},
                mask,
                HOP_LENGTH
            ).unsqueeze(1)
            
            # Calcular métricas
            for i in range(clean_audio.size(0)):
                metrics = calculate_metrics(clean_audio[i], predicted_audio[i])
                for key in test_metrics:
                    test_metrics[key] += metrics[key]
                num_samples += 1
    
    # Calcular métricas médias
    for key in test_metrics:
        test_metrics[key] /= num_samples
    
    print("\nResultados no conjunto de teste:")
    print(f"SNR: {test_metrics['SNR']:.2f} dB")
    print(f"STOI: {test_metrics['STOI']:.4f}")
    
    # Salvar algumas amostras processadas para inspeção auditiva
    os.makedirs('samples', exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 5:  # Salvar 5 amostras
                break
            
            noisy_spec = batch['noisy_spec']['magnitude'].unsqueeze(1).to(DEVICE)
            noisy_audio = batch['noisy_audio'].to(DEVICE)
            clean_audio = batch['clean_audio'].to(DEVICE)
            
            # Forward pass
            mask = model(noisy_spec)
            
            # Reconstruir áudio
            denoised_audio = reconstruct_audio(
                {'magnitude': noisy_spec.squeeze(1), 'phase': batch['noisy_spec']['phase'].to(DEVICE)},
                mask,
                HOP_LENGTH
            )
            
            # Salvar áudios
            torchaudio.save(f'samples/sample_{i}_clean.wav', clean_audio[0].cpu(), SAMPLE_RATE)
            torchaudio.save(f'samples/sample_{i}_noisy.wav', noisy_audio[0].cpu(), SAMPLE_RATE)
            torchaudio.save(f'samples/sample_{i}_denoised.wav', denoised_audio[0].unsqueeze(0).cpu(), SAMPLE_RATE)
    
    # Limpar diretórios temporários
    shutil.rmtree('temp')
    
    print("\nProcesso concluído! Amostras de áudio salvas no diretório 'samples'.")


if __name__ == "__main__":
    main()