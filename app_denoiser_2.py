import os
import sys
import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
import datetime
import shutil
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                            QHBoxLayout, QFileDialog, QLabel, QWidget, QProgressBar,
                            QMessageBox, QDialog, QSlider, QComboBox, QCheckBox,
                            QTabWidget, QSplitter, QFrame, QSpinBox, QDoubleSpinBox,
                            QGroupBox, QRadioButton, QListWidget, QListWidgetItem,
                            QToolBar, QAction, QStatusBar, QMenu, QMenuBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QUrl, QDir
from PyQt5.QtGui import QIcon, QPixmap, QFont, QColor, QPalette
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QAudioRecorder, QAudioEncoderSettings
from PyQt5.QtMultimediaWidgets import QVideoWidget
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Qt5Agg')

# Importar a definição do modelo do arquivo de treinamento
sys.path.append('.')
from treinamento_2 import UNetDenoiser, reconstruct_audio, N_FFT, HOP_LENGTH, SAMPLE_RATE

# Configurações
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'best_denoiser_model.pth'
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "denoiser_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Histórico de processamentos
HISTORY_FILE = os.path.join(OUTPUT_DIR, "history.txt")

# Thread para processamento de áudio em segundo plano
class AudioProcessingThread(QThread):
    finished = pyqtSignal(str, object, object, str)
    progress = pyqtSignal(int, str)
    
    def __init__(self, model, input_file, intensity=1.0):
        super().__init__()
        self.model = model
        self.input_file = input_file
        self.intensity = intensity  # Intensidade do processamento (0.0 a 1.0)
        
    def run(self):
        try:
            # Criar diretório para o processamento atual
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            process_dir = os.path.join(OUTPUT_DIR, f"process_{timestamp}")
            os.makedirs(process_dir, exist_ok=True)
            
            # Copiar arquivo original para o diretório
            original_filename = os.path.basename(self.input_file)
            original_copy_path = os.path.join(process_dir, "original_" + original_filename)
            shutil.copy2(self.input_file, original_copy_path)
            
            # Carregar áudio
            self.progress.emit(10, "Carregando áudio...")
            audio, sr = torchaudio.load(self.input_file)
            
            # Converter para mono se estiver em estéreo
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Reamostrar se necessário
            if sr != SAMPLE_RATE:
                self.progress.emit(15, f"Reamostrando de {sr}Hz para {SAMPLE_RATE}Hz...")
                audio = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(audio)
            
            # Normalizar
            audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
            
            self.progress.emit(20, "Calculando transformada de Fourier...")
            
            # Calcular STFT
            stft = torch.stft(
                audio.squeeze(0), 
                n_fft=N_FFT, 
                hop_length=HOP_LENGTH, 
                window=torch.hann_window(N_FFT), 
                return_complex=True
            )
            
            # Converter para magnitude e fase
            mag = torch.abs(stft)
            phase = torch.angle(stft)
            
            # Aplicar escala logarítmica à magnitude
            log_mag = torch.log1p(mag)
            
            self.progress.emit(30, "Preparando para processamento...")
            
            # Processar em lotes para evitar problemas de memória
            batch_size = 10  # segundos
            samples_per_batch = batch_size * SAMPLE_RATE
            hop_samples = int(samples_per_batch * 0.5)  # 50% de sobreposição
            
            audio_length = audio.shape[1]
            num_batches = max(1, int(np.ceil(audio_length / hop_samples)))
            
            # Preparar áudio processado
            processed_audio = torch.zeros_like(audio)
            
            # Contador para normalização da sobreposição
            overlap_count = torch.zeros_like(audio)
            
            for i in range(num_batches):
                start_sample = i * hop_samples
                end_sample = min(start_sample + samples_per_batch, audio_length)
                
                # Extrair segmento
                segment = audio[:, start_sample:end_sample]
                
                # Calcular STFT do segmento
                segment_stft = torch.stft(
                    segment.squeeze(0), 
                    n_fft=N_FFT, 
                    hop_length=HOP_LENGTH, 
                    window=torch.hann_window(N_FFT), 
                    return_complex=True
                )
                
                segment_mag = torch.abs(segment_stft)
                segment_phase = torch.angle(segment_stft)
                segment_log_mag = torch.log1p(segment_mag)
                
                # Preparar para o modelo
                noisy_spec = segment_log_mag.unsqueeze(0).unsqueeze(0).to(DEVICE)
                
                # Aplicar modelo
                self.progress.emit(40 + int(40 * i / num_batches), 
                                  f"Processando segmento {i+1}/{num_batches}...")
                with torch.no_grad():
                    mask = self.model(noisy_spec)
                
                # Aplicar intensidade ao mask (controle de intensidade do efeito)
                if self.intensity < 1.0:
                    # Interpolar entre máscara original (efeito total) e máscara de 1s (sem efeito)
                    unity_mask = torch.ones_like(mask)
                    mask = self.intensity * mask + (1 - self.intensity) * unity_mask
                
                # Reconstruir áudio
                denoised_segment = reconstruct_audio(
                    {'magnitude': noisy_spec.squeeze(1), 'phase': segment_phase.unsqueeze(0).to(DEVICE)},
                    mask,
                    HOP_LENGTH
                )
                
                # Garantir que denoised_segment tenha a forma correta
                if len(denoised_segment.shape) == 1:
                    denoised_segment = denoised_segment.unsqueeze(0)
                
                # Verificar se o comprimento do segmento é compatível
                if denoised_segment.shape[0] != 1:
                    denoised_segment = denoised_segment.unsqueeze(0)
                
                # Adicionar ao áudio processado com sobreposição
                segment_length = denoised_segment.shape[1]
                if start_sample + segment_length <= processed_audio.shape[1]:
                    processed_audio[0, start_sample:start_sample + segment_length] += denoised_segment[0]
                    overlap_count[0, start_sample:start_sample + segment_length] += 1
                else:
                    # Ajustar se o segmento for maior que o espaço disponível
                    available_length = processed_audio.shape[1] - start_sample
                    processed_audio[0, start_sample:] += denoised_segment[0, :available_length]
                    overlap_count[0, start_sample:] += 1
            
            # Normalizar pela contagem de sobreposição
            processed_audio = processed_audio / (overlap_count + 1e-8)
            
            # Normalizar amplitude
            processed_audio = processed_audio / (torch.max(torch.abs(processed_audio)) + 1e-8)
            
            self.progress.emit(90, "Salvando resultado...")
            
            # Salvar resultado no diretório do processo
            output_file = os.path.join(process_dir, "denoised_" + original_filename)
            torchaudio.save(output_file, processed_audio, SAMPLE_RATE)
            
            # Criar arquivo de informações
            info_file = os.path.join(process_dir, "info.txt")
            with open(info_file, "w") as f:
                f.write(f"Processamento realizado em: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Arquivo original: {self.input_file}\n")
                f.write(f"Duração: {audio_length / SAMPLE_RATE:.2f} segundos\n")
                f.write(f"Taxa de amostragem: {SAMPLE_RATE} Hz\n")
                f.write(f"Dispositivo usado: {DEVICE}\n")
                f.write(f"Intensidade do processamento: {self.intensity:.2f}\n")
            
            # Adicionar ao histórico
            with open(HISTORY_FILE, "a") as f:
                f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{self.input_file},{output_file},{process_dir}\n")
            
            self.progress.emit(100, "Processamento concluído!")
            
            # Emitir sinal de conclusão
            self.finished.emit(output_file, audio.numpy(), processed_audio.numpy(), process_dir)
            
        except Exception as e:
            print(f"Erro no processamento: {e}")
            import traceback
            traceback.print_exc()
            self.progress.emit(0, f"Erro: {str(e)}")
            self.finished.emit(None, None, None, None)


# Classe para o canvas de visualização de forma de onda
class WaveformCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(WaveformCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()
        
    def plot_waveform(self, audio_data=None, title="Forma de Onda"):
        self.axes.clear()
        if audio_data is not None:
            time = np.arange(0, len(audio_data)) / SAMPLE_RATE
            self.axes.plot(time, audio_data, linewidth=0.5)
            self.axes.set_xlabel('Tempo (s)')
            self.axes.set_ylabel('Amplitude')
        self.axes.set_title(title)
        self.fig.tight_layout()
        self.draw()
    
    def plot_spectrogram(self, audio_data=None, title="Espectrograma"):
        self.axes.clear()
        if audio_data is not None:
            D = librosa.amplitude_to_db(
                np.abs(librosa.stft(audio_data, n_fft=N_FFT, hop_length=HOP_LENGTH)),
                ref=np.max
            )
            librosa.display.specshow(
                D, 
                sr=SAMPLE_RATE, 
                hop_length=HOP_LENGTH, 
                x_axis='time', 
                y_axis='log',
                ax=self.axes
            )
            self.axes.set_title(title)
        else:
            self.axes.set_title(title)
        self.fig.tight_layout()
        self.draw()


# Diálogo para gravação de áudio
class AudioRecordDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gravar Áudio")
        self.setGeometry(300, 300, 500, 300)
        
        # Configurar gravador de áudio
        self.recorder = QAudioRecorder()
        settings = QAudioEncoderSettings()
        settings.setCodec("audio/pcm")
        settings.setSampleRate(SAMPLE_RATE)
        settings.setChannelCount(1)
        settings.setQuality(2)  # 0=VeryLow, 1=Low, 2=Normal, 3=High, 4=VeryHigh
        self.recorder.setEncodingSettings(settings)
        
        # Definir diretório de saída
        self.output_file = os.path.join(OUTPUT_DIR, f"recorded_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
        self.recorder.setOutputLocation(QUrl.fromLocalFile(self.output_file))
        
        # Layout
        layout = QVBoxLayout()
        
        # Título
        title_label = QLabel("Gravação de Áudio")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Separador
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # Label de status
        self.status_label = QLabel("Pronto para gravar")
        self.status_label.setAlignment(Qt.AlignCenter)
        status_font = QFont()
        status_font.setPointSize(12)
        self.status_label.setFont(status_font)
        layout.addWidget(self.status_label)
        
        # Visualização de nível de áudio
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setValue(0)
        layout.addWidget(self.level_bar)
        
        # Timer para mostrar duração da gravação e nível de áudio
        self.timer = QTimer()
        self.timer.setInterval(100)  # Atualizar a cada 100ms
        self.timer.timeout.connect(self.update_ui)
        self.duration = 0
        
        # Botões
        button_layout = QHBoxLayout()
        
        self.record_button = QPushButton("Gravar")
        self.record_button.setIcon(QIcon.fromTheme("media-record"))
        self.record_button.clicked.connect(self.toggle_recording)
        button_layout.addWidget(self.record_button)
        
        self.stop_button = QPushButton("Parar")
        self.stop_button.setIcon(QIcon.fromTheme("media-playback-stop"))
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        layout.addLayout(button_layout)
        
        # Botão para usar o áudio gravado
        self.use_button = QPushButton("Usar Áudio Gravado")
        self.use_button.setIcon(QIcon.fromTheme("dialog-ok-apply"))
        self.use_button.clicked.connect(self.accept)
        self.use_button.setEnabled(False)
        layout.addWidget(self.use_button)
        
        # Informações
        info_label = QLabel("O áudio será gravado em formato WAV mono com taxa de amostragem de 16kHz.")
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
        
        self.setLayout(layout)
        
    def toggle_recording(self):
        if self.recorder.state() == QAudioRecorder.StoppedState:
            self.recorder.record()
            self.record_button.setText("Pausar")
            self.stop_button.setEnabled(True)
            self.status_label.setText("Gravando...")
            self.duration = 0
            self.timer.start()
        elif self.recorder.state() == QAudioRecorder.RecordingState:
            self.recorder.pause()
            self.record_button.setText("Continuar")
            self.status_label.setText("Gravação pausada")
            self.timer.stop()
        elif self.recorder.state() == QAudioRecorder.PausedState:
            self.recorder.record()
            self.record_button.setText("Pausar")
            self.status_label.setText("Gravando...")
            self.timer.start()
    
    def stop_recording(self):
        if self.recorder.state() != QAudioRecorder.StoppedState:
            self.recorder.stop()
            self.record_button.setText("Gravar")
            self.stop_button.setEnabled(False)
            self.status_label.setText("Gravação concluída")
            self.use_button.setEnabled(True)
            self.timer.stop()
    
    def update_ui(self):
        # Atualizar duração
        self.duration += 0.1
        minutes = int(self.duration) // 60
        seconds = int(self.duration) % 60
        ms = int((self.duration - int(self.duration)) * 10)
        self.status_label.setText(f"Gravando... {minutes:02d}:{seconds:02d}.{ms}")
        
        # Simular nível de áudio (em uma aplicação real, usaríamos o nível real)
        import random
        level = random.randint(10, 80)
        if self.recorder.state() == QAudioRecorder.RecordingState:
            self.level_bar.setValue(level)
    
    def get_recorded_file(self):
        return self.output_file


# Diálogo de configurações
class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configurações")
        self.setGeometry(300, 300, 400, 300)
        
        layout = QVBoxLayout()
        
        # Grupo de configurações de áudio
        audio_group = QGroupBox("Configurações de Áudio")
        audio_layout = QVBoxLayout()
        
        # Taxa de amostragem
        sample_rate_layout = QHBoxLayout()
        sample_rate_layout.addWidget(QLabel("Taxa de amostragem:"))
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems(["16000 Hz", "22050 Hz", "44100 Hz", "48000 Hz"])
        self.sample_rate_combo.setCurrentIndex(0)  # 16000 Hz por padrão
        sample_rate_layout.addWidget(self.sample_rate_combo)
        audio_layout.addLayout(sample_rate_layout)
        
        # Tamanho da FFT
        fft_layout = QHBoxLayout()
        fft_layout.addWidget(QLabel("Tamanho da FFT:"))
        self.fft_combo = QComboBox()
        self.fft_combo.addItems(["256", "512", "1024", "2048"])
        self.fft_combo.setCurrentIndex(1)  # 512 por padrão
        fft_layout.addWidget(self.fft_combo)
        audio_layout.addLayout(fft_layout)
        
        audio_group.setLayout(audio_layout)
        layout.addWidget(audio_group)
        
        # Grupo de configurações de interface
        ui_group = QGroupBox("Interface")
        ui_layout = QVBoxLayout()
        
        # Tema
        theme_layout = QHBoxLayout()
        theme_layout.addWidget(QLabel("Tema:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Claro", "Escuro", "Sistema"])
        self.theme_combo.setCurrentIndex(2)  # Sistema por padrão
        theme_layout.addWidget(self.theme_combo)
        ui_layout.addLayout(theme_layout)
        
        # Visualização
        self.show_spectrogram = QCheckBox("Mostrar espectrograma")
        self.show_spectrogram.setChecked(True)
        ui_layout.addWidget(self.show_spectrogram)
        
        ui_group.setLayout(ui_layout)
        layout.addWidget(ui_group)
        
        # Botões
        button_layout = QHBoxLayout()
        save_button = QPushButton("Salvar")
        save_button.clicked.connect(self.accept)
        button_layout.addWidget(save_button)
        
        cancel_button = QPushButton("Cancelar")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def get_settings(self):
        sample_rates = [16000, 22050, 44100, 48000]
        fft_sizes = [256, 512, 1024, 2048]
        themes = ["light", "dark", "system"]
        
        return {
            "sample_rate": sample_rates[self.sample_rate_combo.currentIndex()],
            "fft_size": fft_sizes[self.fft_combo.currentIndex()],
            "theme": themes[self.theme_combo.currentIndex()],
            "show_spectrogram": self.show_spectrogram.isChecked()
        }


# Diálogo de histórico
class HistoryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Histórico de Processamentos")
        self.setGeometry(300, 300, 700, 500)
        
        layout = QVBoxLayout()
        
        # Lista de processamentos
        self.history_list = QListWidget()
        self.history_list.itemDoubleClicked.connect(self.open_process_dir)
        layout.addWidget(self.history_list)
        
        # Botões
        button_layout = QHBoxLayout()
        
        open_button = QPushButton("Abrir Diretório")
        open_button.clicked.connect(self.open_selected_dir)
        button_layout.addWidget(open_button)
        
        play_button = QPushButton("Reproduzir")
        play_button.clicked.connect(self.play_selected)
        button_layout.addWidget(play_button)
        
        clear_button = QPushButton("Limpar Histórico")
        clear_button.clicked.connect(self.clear_history)
        button_layout.addWidget(clear_button)
        
        close_button = QPushButton("Fechar")
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # Carregar histórico
        self.load_history()
    
    def load_history(self):
        self.history_list.clear()
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split(",")
                        if len(parts) >= 4:
                            timestamp, input_file, output_file, process_dir = parts[:4]
                            item = QListWidgetItem(f"{timestamp} - {os.path.basename(input_file)}")
                            item.setData(Qt.UserRole, process_dir)
                            self.history_list.addItem(item)
    
    def open_process_dir(self, item):
        process_dir = item.data(Qt.UserRole)
        if process_dir and os.path.exists(process_dir):
            self.open_directory(process_dir)
    
    def open_selected_dir(self):
        selected_items = self.history_list.selectedItems()
        if selected_items:
            process_dir = selected_items[0].data(Qt.UserRole)
            if process_dir and os.path.exists(process_dir):
                self.open_directory(process_dir)
    
    def play_selected(self):
        selected_items = self.history_list.selectedItems()
        if selected_items:
            process_dir = selected_items[0].data(Qt.UserRole)
            if process_dir and os.path.exists(process_dir):
                # Encontrar arquivo processado
                for file in os.listdir(process_dir):
                    if file.startswith("denoised_") and file.endswith((".wav", ".mp3", ".ogg", ".flac")):
                        output_file = os.path.join(process_dir, file)
                        # Emitir sinal para reproduzir (implementado pelo pai)
                        self.parent().play_file(output_file)
                        break
    
    def clear_history(self):
        reply = QMessageBox.question(
            self, 
            "Limpar Histórico", 
            "Tem certeza que deseja limpar o histórico? Isso não excluirá os arquivos processados.",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            self.history_list.clear()
    
    def open_directory(self, path):
        import subprocess
        import platform
        
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.call(["open", path])
        else:  # Linux
            subprocess.call(["xdg-open", path])


# Classe principal da aplicação
class AudioDenoisingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Aplicação de Supressão de Ruído")
        self.setGeometry(100, 100, 1200, 800)
        
        # Carregar modelo
        self.model = UNetDenoiser().to(DEVICE)
        try:
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            self.model.eval()
            print("Modelo carregado com sucesso!")
        except Exception as e:
            print(f"Erro ao carregar o modelo: {e}")
            QMessageBox.warning(
                self, 
                "Erro ao Carregar Modelo", 
                f"Não foi possível carregar o modelo: {str(e)}\n\nVerifique se o arquivo {MODEL_PATH} existe."
            )
        
        # Variáveis para armazenar caminhos de arquivos
        self.input_file = None
        self.output_file = None
        self.current_process_dir = None
        
        # Players de áudio
        self.player_original = QMediaPlayer()
        self.player_denoised = QMediaPlayer()
        
        # Configurações
        self.settings = {
            "sample_rate": SAMPLE_RATE,
            "fft_size": N_FFT,
            "theme": "system",
            "show_spectrogram": True,
            "processing_intensity": 1.0
        }
        
        # Configurar interface
        self.setup_ui()
        
        # Verificar histórico
        if not os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "w") as f:
                f.write("# Histórico de processamentos\n")
                f.write("# Formato: timestamp,input_file,output_file,process_dir\n")
    
    def setup_ui(self):
        # Barra de status
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Pronto")
        
        # Barra de menu
        menubar = self.menuBar()
        
        # Menu Arquivo
        file_menu = menubar.addMenu("Arquivo")
        
        open_action = QAction("Abrir Áudio", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.load_audio)
        file_menu.addAction(open_action)
        
        record_action = QAction("Gravar Áudio", self)
        record_action.setShortcut("Ctrl+R")
        record_action.triggered.connect(self.record_audio)
        file_menu.addAction(record_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Sair", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Menu Processamento
        process_menu = menubar.addMenu("Processamento")
        
        process_action = QAction("Processar Áudio", self)
        process_action.setShortcut("Ctrl+P")
        process_action.triggered.connect(self.process_audio)
        process_menu.addAction(process_action)
        
        history_action = QAction("Histórico", self)
        history_action.setShortcut("Ctrl+H")
        history_action.triggered.connect(self.show_history)
        process_menu.addAction(history_action)
        
        # Menu Configurações
        settings_menu = menubar.addMenu("Configurações")
        
        settings_action = QAction("Preferências", self)
        settings_action.triggered.connect(self.show_settings)
        settings_menu.addAction(settings_action)
        
        # Menu Ajuda
        help_menu = menubar.addMenu("Ajuda")
        
        about_action = QAction("Sobre", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        # Widget principal
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Layout principal
        main_layout = QVBoxLayout()
        
        # Área de título
        title_label = QLabel("Supressão de Ruído com Aprendizado Profundo")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Área de informações do arquivo
        self.file_label = QLabel("Nenhum arquivo carregado")
        self.file_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.file_label)
        
        # Botões de ação
        button_layout = QHBoxLayout()
        
        self.load_button = QPushButton("Carregar Áudio")
        self.load_button.clicked.connect(self.load_audio)
        button_layout.addWidget(self.load_button)
        
        self.record_button = QPushButton("Gravar Áudio")
        self.record_button.clicked.connect(self.record_audio)
        button_layout.addWidget(self.record_button)
        
        self.process_button = QPushButton("Processar")
        self.process_button.clicked.connect(self.process_audio)
        self.process_button.setEnabled(False)
        button_layout.addWidget(self.process_button)
        
        self.open_dir_button = QPushButton("Abrir Diretório de Saída")
        self.open_dir_button.clicked.connect(self.open_output_dir)
        button_layout.addWidget(self.open_dir_button)
        
        main_layout.addLayout(button_layout)
        
        # Controle de intensidade
        intensity_layout = QHBoxLayout()
        intensity_layout.addWidget(QLabel("Intensidade do Processamento:"))
        
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setRange(0, 100)
        self.intensity_slider.setValue(100)
        self.intensity_slider.setTickPosition(QSlider.TicksBelow)
        self.intensity_slider.setTickInterval(10)
        intensity_layout.addWidget(self.intensity_slider)
        
        self.intensity_label = QLabel("100%")
        intensity_layout.addWidget(self.intensity_label)
        
        self.intensity_slider.valueChanged.connect(self.update_intensity)
        
        main_layout.addLayout(intensity_layout)
        
        # Barra de progresso
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        # Label de status do processamento
        self.processing_status = QLabel("")
        self.processing_status.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.processing_status)
        
        # Área de visualização de forma de onda
        waveform_layout = QHBoxLayout()
        
        # Canvas para áudio original
        self.original_canvas = WaveformCanvas(self, width=5, height=2)
        self.original_canvas.plot_waveform(title="Áudio Original")
        waveform_layout.addWidget(self.original_canvas)
        
        # Canvas para áudio processado
        self.processed_canvas = WaveformCanvas(self, width=5, height=2)
        self.processed_canvas.plot_waveform(title="Áudio Processado")
        waveform_layout.addWidget(self.processed_canvas)
        
        main_layout.addLayout(waveform_layout)
        
        # Área de controles de reprodução
        playback_layout = QHBoxLayout()
        
        # Controles para áudio original
        original_playback = QVBoxLayout()
        original_group = QGroupBox("Áudio Original")
        original_controls = QVBoxLayout()
        
        self.play_original_button = QPushButton("Reproduzir")
        self.play_original_button.clicked.connect(self.play_original)
        self.play_original_button.setEnabled(False)
        original_controls.addWidget(self.play_original_button)
        
        self.stop_original_button = QPushButton("Parar")
        self.stop_original_button.clicked.connect(self.stop_original)
        self.stop_original_button.setEnabled(False)
        original_controls.addWidget(self.stop_original_button)
        
        original_group.setLayout(original_controls)
        playback_layout.addWidget(original_group)
        
        # Controles para áudio processado
        processed_playback = QVBoxLayout()
        processed_group = QGroupBox("Áudio Processado")
        processed_controls = QVBoxLayout()
        
        self.play_processed_button = QPushButton("Reproduzir")
        self.play_processed_button.clicked.connect(self.play_processed)
        self.play_processed_button.setEnabled(False)
        processed_controls.addWidget(self.play_processed_button)
        
        self.stop_processed_button = QPushButton("Parar")
        self.stop_processed_button.clicked.connect(self.stop_processed)
        self.stop_processed_button.setEnabled(False)
        processed_controls.addWidget(self.stop_processed_button)
        
        processed_group.setLayout(processed_controls)
        playback_layout.addWidget(processed_group)
        
        main_layout.addLayout(playback_layout)
        
        # Definir layout principal
        main_widget.setLayout(main_layout)
    
    def update_intensity(self):
        value = self.intensity_slider.value()
        self.intensity_label.setText(f"{value}%")
        self.settings["processing_intensity"] = value / 100.0
    
    def load_audio(self):
        file_dialog = QFileDialog()
        self.input_file, _ = file_dialog.getOpenFileName(
            self, "Selecionar Arquivo de Áudio", "", "Arquivos de Áudio (*.wav *.mp3 *.ogg *.flac)"
        )
        
        if self.input_file:
            self.file_label.setText(f"Arquivo carregado: {os.path.basename(self.input_file)}")
            self.process_button.setEnabled(True)
            
            # Carregar e exibir forma de onda
            try:
                audio, sr = librosa.load(self.input_file, sr=None)
                self.original_canvas.plot_waveform(audio, "Áudio Original")
                self.processed_canvas.plot_waveform(title="Áudio Processado (aguardando processamento)")
                
                # Configurar player original
                self.player_original.setMedia(QMediaContent(QUrl.fromLocalFile(self.input_file)))
                self.play_original_button.setEnabled(True)
                self.stop_original_button.setEnabled(True)
                
                # Atualizar status
                duration = librosa.get_duration(y=audio, sr=sr)
                self.statusBar.showMessage(f"Arquivo carregado: {os.path.basename(self.input_file)} | Duração: {int(duration//60)}:{int(duration%60):02d} | Taxa: {sr} Hz")
            except Exception as e:
                print(f"Erro ao carregar o áudio: {e}")
                QMessageBox.warning(self, "Erro", f"Erro ao carregar o áudio: {str(e)}")
    
    def record_audio(self):
        dialog = AudioRecordDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self.input_file = dialog.get_recorded_file()
            if os.path.exists(self.input_file):
                self.file_label.setText(f"Áudio gravado: {os.path.basename(self.input_file)}")
                self.process_button.setEnabled(True)
                
                # Carregar e exibir forma de onda
                try:
                    audio, sr = librosa.load(self.input_file, sr=None)
                    self.original_canvas.plot_waveform(audio, "Áudio Original (Gravado)")
                    self.processed_canvas.plot_waveform(title="Áudio Processado (aguardando processamento)")
                    
                    # Configurar player original
                    self.player_original.setMedia(QMediaContent(QUrl.fromLocalFile(self.input_file)))
                    self.play_original_button.setEnabled(True)
                    self.stop_original_button.setEnabled(True)
                    
                    # Atualizar status
                    duration = librosa.get_duration(y=audio, sr=sr)
                    self.statusBar.showMessage(f"Áudio gravado | Duração: {int(duration//60)}:{int(duration%60):02d} | Taxa: {sr} Hz")
                except Exception as e:
                    print(f"Erro ao carregar o áudio gravado: {e}")
                    QMessageBox.warning(self, "Erro", f"Erro ao carregar o áudio gravado: {str(e)}")
    
    def process_audio(self):
        if not self.input_file:
            return
        
        # Desabilitar botões durante o processamento
        self.load_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.process_button.setEnabled(False)
        self.play_original_button.setEnabled(False)
        self.play_processed_button.setEnabled(False)
        
        # Iniciar thread de processamento
        intensity = self.settings["processing_intensity"]
        self.processing_thread = AudioProcessingThread(self.model, self.input_file, intensity)
        self.processing_thread.progress.connect(self.update_progress)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.start()
    
    def update_progress(self, value, status_text):
        self.progress_bar.setValue(value)
        self.processing_status.setText(status_text)
    
    def processing_finished(self, output_file, original_audio, processed_audio, process_dir):
        # Reabilitar botões
        self.load_button.setEnabled(True)
        self.record_button.setEnabled(True)
        self.process_button.setEnabled(True)
        self.play_original_button.setEnabled(True)
        
        if output_file:
            self.output_file = output_file
            self.current_process_dir = process_dir
            self.file_label.setText(f"Arquivo processado: {os.path.basename(output_file)}")
            
            # Exibir forma de onda processada
            self.processed_canvas.plot_waveform(processed_audio.squeeze(), "Áudio Processado")
            
            # Configurar player processado
            self.player_denoised.setMedia(QMediaContent(QUrl.fromLocalFile(output_file)))
            self.play_processed_button.setEnabled(True)
            self.stop_processed_button.setEnabled(True)
            
            # Mostrar mensagem de sucesso
            QMessageBox.information(self, "Processamento Concluído", 
                                   f"Áudio processado com sucesso!\nSalvo em: {output_file}")
            
            # Atualizar status
            self.statusBar.showMessage(f"Processamento concluído | Arquivo salvo em: {os.path.basename(output_file)}")
        else:
            self.file_label.setText("Erro no processamento. Tente novamente.")
            self.processing_status.setText("")
            QMessageBox.warning(self, "Erro", "Ocorreu um erro durante o processamento do áudio.")
            self.statusBar.showMessage("Erro no processamento")
    
    def play_original(self):
        self.player_original.play()
        self.statusBar.showMessage("Reproduzindo áudio original...")
    
    def stop_original(self):
        self.player_original.stop()
        self.statusBar.showMessage("Reprodução interrompida")
    
    def play_processed(self):
        self.player_denoised.play()
        self.statusBar.showMessage("Reproduzindo áudio processado...")
    
    def stop_processed(self):
        self.player_denoised.stop()
        self.statusBar.showMessage("Reprodução interrompida")
    
    def open_output_dir(self):
        import subprocess
        import platform
        
        if platform.system() == "Windows":
            os.startfile(OUTPUT_DIR)
        elif platform.system() == "Darwin":  # macOS
            subprocess.call(["open", OUTPUT_DIR])
        else:  # Linux
            subprocess.call(["xdg-open", OUTPUT_DIR])
    
    def show_history(self):
        dialog = HistoryDialog(self)
        dialog.exec_()
    
    def show_settings(self):
        dialog = SettingsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            new_settings = dialog.get_settings()
            self.settings.update(new_settings)
            
            # Aplicar tema
            if new_settings["theme"] == "dark":
                self.apply_dark_theme()
            elif new_settings["theme"] == "light":
                self.apply_light_theme()
            # Para "system", mantemos o tema padrão
    
    def show_about(self):
        QMessageBox.about(self, "Sobre", 
                         "Aplicação de Supressão de Ruído\n\n"
                         "Desenvolvida com PyQt5 e PyTorch\n\n"
                         "Esta aplicação utiliza um modelo de rede neural U-Net para remover ruído de arquivos de áudio.")
    
    def apply_dark_theme(self):
        # Aplicar tema escuro
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(palette)
    
    def apply_light_theme(self):
        # Aplicar tema claro (padrão)
        self.setPalette(QApplication.style().standardPalette())
    
    def play_file(self, file_path):
        # Método para reproduzir um arquivo a partir do diálogo de histórico
        self.player_denoised.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
        self.player_denoised.play()
        self.statusBar.showMessage(f"Reproduzindo: {os.path.basename(file_path)}")


# Iniciar aplicação
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioDenoisingApp()
    window.show()
    sys.exit(app.exec_())