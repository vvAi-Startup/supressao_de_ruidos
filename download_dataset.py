from datasets import load_dataset
import os
import soundfile as sf
 
# Carregar o dataset
dataset = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")
 
# Criar diretórios se não existirem
os.makedirs("datasets/clean", exist_ok=True)
os.makedirs("datasets/noisy", exist_ok=True)
 
# Salvar arquivos de áudio
for i, sample in enumerate(dataset["train"]):
    clean_path = f"datasets/clean/audio{i}.wav"
    noisy_path = f"datasets/noisy/audio{i}.wav"
    sf.write(clean_path, sample["clean"]["array"], sample["clean"]["sampling_rate"])
    sf.write(noisy_path, sample["noisy"]["array"], sample["noisy"]["sampling_rate"])