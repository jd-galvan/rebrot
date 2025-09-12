import os
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch
import pyiqa
import lpips

# === Carpetas ===
carpeta_daniadas = '/home/salvem/jgalvan/LPIPS/originales/'        # Cambia esto según tu estructura
carpeta_restauradas = '/home/salvem/jgalvan/LPIPS/restauradas/'

# === Transformaciones ===
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# === Inicializar métricas ===
#niqe_metric = pyiqa.create_metric('niqe')
#brisque_metric = pyiqa.create_metric('brisque')
lpips_metric = lpips.LPIPS(net='vgg')

print("LLEGO AQUI")

# === Listado de archivos ===
daniadas_files = [f for f in os.listdir(carpeta_daniadas) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
restauradas_files = [f for f in os.listdir(carpeta_restauradas) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# === Emparejar por nombre base ===
def buscar_restaurada(nombre_base, lista_restauradas):
    coincidencias = [f for f in lista_restauradas if f.startswith(nombre_base)]
    return coincidencias[0] if coincidencias else None

# === Diagnóstico automático ===
def diagnosticar(delta_niqe, delta_brisque, lpips_val):
    if delta_niqe < -0.1 and lpips_val > 0.1:
        return "✔ Mejoró y restauró bien"
    elif delta_niqe > 0.3 and lpips_val < 0.05:
        return "⚠ No mejoró ni restauró"
    elif delta_niqe > 0.3 and lpips_val > 0.2 and delta_brisque > 10:
        return "⚠ Posible artefacto añadido"
    else:
        return "🟡 Revisión manual sugerida"

# === Evaluación ===
resultados = []

for archivo_daniado in daniadas_files:
    nombre_base = os.path.splitext(archivo_daniado)[0]
    archivo_restaurado = buscar_restaurada(nombre_base, restauradas_files)

    if archivo_restaurado is None:
        print(f"No se encontró imagen restaurada para {archivo_daniado}")
        continue

    try:
        ruta_daniada = os.path.join(carpeta_daniadas, archivo_daniado)
        ruta_restaurada = os.path.join(carpeta_restauradas, archivo_restaurado)

        img_daniada = Image.open(ruta_daniada).convert('RGB')
        img_restaurada = Image.open(ruta_restaurada).convert('RGB')

        tensor_daniada = transform(img_daniada).unsqueeze(0)
        tensor_restaurada = transform(img_restaurada).unsqueeze(0)

        #niqe_dan = niqe_metric(tensor_daniada).item()
        #niqe_res = niqe_metric(tensor_restaurada).item()
        #bris_dan = brisque_metric(tensor_daniada).item()
        #bris_res = brisque_metric(tensor_restaurada).item()

        lpips_val = lpips_metric(
            tensor_daniada * 2 - 1,
            tensor_restaurada * 2 - 1
        ).item()

        #delta_niqe = niqe_res - niqe_dan
        #delta_brisque = bris_res - bris_dan
        #diagnostico = diagnosticar(delta_niqe, delta_brisque, lpips_val)

        resultados.append({
            'imagen_daniada': archivo_daniado,
            'imagen_restaurada': archivo_restaurado,
            #'NIQE_daniada': round(niqe_dan, 4),
            #'NIQE_restaurada': round(niqe_res, 4),
            #'delta_NIQE': round(delta_niqe, 4),
            #'BRISQUE_daniada': round(bris_dan, 4),
            #'BRISQUE_restaurada': round(bris_res, 4),
            #'delta_BRISQUE': round(delta_brisque, 4),
            'LPIPS_vs_daniada': round(lpips_val, 4),
            #'diagnostico': diagnostico
        })

    except Exception as e:
        print(f"❌ Error procesando {archivo_daniado}: {e}")

# === Guardar CSV final ===
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("evaluacion_completa_600.csv", index=False)
print("✅ Evaluación completa guardada como 'evaluacion_completa.csv'")
