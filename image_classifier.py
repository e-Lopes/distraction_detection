import os
import shutil
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# === Selecionar pasta de origem ===
#origem = filedialog.askdirectory(title="Selecione a pasta com imagens .jpg/.jpeg")
origem = '/home/edu/Desktop/distraction_detection/distraction_detection/frames'
# Criar subpastas OK e NOK
ok_dir = os.path.join(origem, "OK")
nok_dir = os.path.join(origem, "NOK")
os.makedirs(ok_dir, exist_ok=True)
os.makedirs(nok_dir, exist_ok=True)

# Lista de imagens válidas
extensoes_validas = ('.jpeg', '.jpg')
imagens = [f for f in os.listdir(origem)
           if f.lower().endswith(extensoes_validas)
           and os.path.isfile(os.path.join(origem, f))]
print(f"Quantidade de imagens encontradas: {len(imagens)}")

# Contar imagens já classificadas
ok_count = len([f for f in os.listdir(ok_dir) if f.lower().endswith(extensoes_validas)])
nok_count = len([f for f in os.listdir(nok_dir) if f.lower().endswith(extensoes_validas)])
print(f"Já classificadas - OK: {ok_count} | NOK: {nok_count}")

indice = 0

# === Janela ===
janela = tk.Tk()
janela.title("Classificador de Imagens")

# Área de imagem
label_imagem = tk.Label(janela)
label_imagem.pack()

def carregar_imagem():
    if indice < len(imagens):
        caminho = os.path.join(origem, imagens[indice])
        try:
            imagem = Image.open(caminho)
            imagem.thumbnail((800, 600))
            foto = ImageTk.PhotoImage(imagem)
            label_imagem.config(image=foto)
            label_imagem.image = foto
            janela.title(f"Imagem: {imagens[indice]}")
        except Exception as e:
            print(f"Erro ao abrir imagem {imagens[indice]}: {e}")
            pular_imagem()
    else:
        label_imagem.config(text="✅ Todas as imagens foram classificadas!")
        botao_ok.config(state=tk.DISABLED)
        botao_nok.config(state=tk.DISABLED)

def classificar(ok=True):
    global indice
    origem_arquivo = os.path.join(origem, imagens[indice])
    destino = os.path.join(ok_dir if ok else nok_dir, imagens[indice])
    shutil.move(origem_arquivo, destino)
    
    # Exibir no terminal
    status = "OK" if ok else "NOK"
    print(f"[{status}] Moved: {imagens[indice]} → {destino}")

    indice += 1
    carregar_imagem()

def pular_imagem():
    global indice
    indice += 1
    carregar_imagem()

# === Teclas de atalho ===
def tecla_pressionada(event):
    if event.char == '1':
        classificar(True)
    elif event.char == '2':
        classificar(False)

janela.bind('<Key>', tecla_pressionada)

# === Botões ===
frame_botoes = tk.Frame(janela)
frame_botoes.pack(pady=10)

botao_ok = tk.Button(frame_botoes, text="✅ OK (1)", width=20, command=lambda: classificar(True))
botao_ok.grid(row=0, column=0, padx=10)

botao_nok = tk.Button(frame_botoes, text="❌ NOK (2)", width=20, command=lambda: classificar(False))
botao_nok.grid(row=0, column=1, padx=10)

# === Início ===
carregar_imagem()
janela.mainloop()
