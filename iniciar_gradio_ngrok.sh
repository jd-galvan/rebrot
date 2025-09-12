#!/bin/bash

# Lanza la app de Gradio en segundo plano
nohup python3 main_yolo.py > salida.log 2>&1 &

# Lanza ngrok en segundo plano
nohup ngrok http 7860 > ngrok.log 2>&1 &

# Espera unos segundos para que ngrok se inicie
sleep 5

# Extrae la URL de ngrok desde su API local
URL=$(curl -s http://127.0.0.1:4040/api/tunnels | jq -r '.tunnels[0].public_url')

echo "🔗 Tu app Gradio está disponible en: $URL"
