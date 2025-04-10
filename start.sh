#!/bin/bash

# Iniciar servidor X virtual se não houver display disponível
if [ -z "$DISPLAY" ] || [ "$DISPLAY" = ":99" ]; then
    echo "Iniciando servidor X virtual..."
    Xvfb :99 -screen 0 1024x768x16 &
    export DISPLAY=:99
    
    # Verificar se o Xvfb iniciou corretamente
    sleep 2
    if ! ps aux | grep -v grep | grep Xvfb > /dev/null; then
        echo "ERRO: Falha ao iniciar o servidor X virtual"
        exit 1
    fi
    echo "Servidor X virtual iniciado com sucesso"
fi

# Configurar e iniciar PulseAudio
mkdir -p /tmp/pulse
echo "Configurando PulseAudio..."
pulseaudio --start --log-target=syslog --system --disallow-exit

# Aguardar PulseAudio iniciar
sleep 2
if ! ps aux | grep -v grep | grep pulseaudio > /dev/null; then
    echo "ERRO: Falha ao iniciar o PulseAudio"
    exit 1
fi
echo "PulseAudio iniciado com sucesso"

# Listar variáveis de ambiente relacionadas ao display
echo "Variáveis de ambiente de display:"
echo "DISPLAY=$DISPLAY"
echo "XAUTHORITY=$XAUTHORITY"

# Verificar se o diretório XCB existe
echo "Verificando plugins Qt:"
find /usr/local/lib/python3.9/site-packages/PyQt5/Qt5/plugins -name "*.so" | sort

# Executar a aplicação
echo "Iniciando aplicação de supressão de ruído..."
python app_denoiser.py 