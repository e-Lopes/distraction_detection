import os
import cv2
import csv
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from collections import defaultdict, Counter
from sklearn.metrics import classification_report, confusion_matrix

# Configurações
FRAMES_DIR = "frames_por_estado"
OUTPUT_CSV = "validacao_manual.csv"
STATES_TO_VALIDAR = ["NORMAL", "CELULAR", "POSTURA_RUIM", "MAOS_FORA", "NAO_DETECTADO"]

TECLA_ESTADO = {
    "1": "NORMAL",
    "2": "CELULAR",
    "3": "POSTURA_RUIM",
    "4": "MAOS_FORA",
    "5": "NAO_DETECTADO",
    "0": "IGNORAR"
}

class ValidadorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Validação Manual de Frames")
        self.label_img = tk.Label(root)
        self.label_img.pack()

        for tecla, estado in TECLA_ESTADO.items():
            if estado != "IGNORAR":
                btn = tk.Button(root, text=f"{tecla} - {estado}", width=20,
                                command=lambda e=estado: self.registrar_rotulo(e))
                btn.pack(pady=2)

        self.root.bind("<Key>", self.tecla_pressionada)

        self.limpar_csv()
        self.registros_existentes = self.carregar_csv_existente()
        self.frames = self.carregar_frames_nao_anotados()

        if not self.frames:
            messagebox.showinfo("Tudo pronto", "Todos os frames já foram anotados!")
            self.mostrar_estatisticas()
            self.root.quit()
            return

        self.index_atual = 0
        self.registros_novos = []
        self.mostrar_proximo_frame()

    def limpar_csv(self):
        if not os.path.exists(OUTPUT_CSV):
            return

        linhas_unicas = {}
        with open(OUTPUT_CSV, newline="") as f:
            reader = csv.reader(f)
            cabecalho = next(reader, None)
            for row in reader:
                if len(row) != 3 or not all(c.strip() for c in row):
                    continue
                chave = (row[0], row[1])  # arquivo + estado_predito
                linhas_unicas[chave] = row

        with open(OUTPUT_CSV, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["arquivo", "estado_predito", "estado_real"])
            for row in linhas_unicas.values():
                writer.writerow(row)

    def salvar_linha_csv(self, linha):
        with open(OUTPUT_CSV, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(linha)

    def carregar_csv_existente(self):
        anotados = set()
        if os.path.exists(OUTPUT_CSV):
            with open(OUTPUT_CSV, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["arquivo"].strip():
                        anotados.add((row["estado_predito"], row["arquivo"]))
        return anotados

    def carregar_frames_nao_anotados(self):
        frames = []
        for estado_predito in STATES_TO_VALIDAR:
            pasta = os.path.join(FRAMES_DIR, estado_predito)
            if not os.path.isdir(pasta):
                continue
            arquivos = sorted(f for f in os.listdir(pasta) if f.endswith(".jpg"))
            for nome in arquivos:
                if (estado_predito, nome) not in self.registros_existentes:
                    caminho = os.path.join(pasta, nome)
                    frames.append((caminho, estado_predito, nome))
        return frames

    def mostrar_proximo_frame(self):
        if self.index_atual >= len(self.frames):
            self.salvar_csv()
            self.mostrar_estatisticas()
            messagebox.showinfo("Fim", "Validação finalizada!")
            self.root.quit()
            return

        caminho, estado_predito, nome = self.frames[self.index_atual]
        imagem = cv2.imread(caminho)
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        imagem = cv2.resize(imagem, (640, 480))
        imagem_pil = Image.fromarray(imagem)
        imagem_tk = ImageTk.PhotoImage(imagem_pil)

        self.label_img.configure(image=imagem_tk)
        self.label_img.image = imagem_tk
        self.root.title(f"{nome} | Predito: {estado_predito} | Frame {self.index_atual + 1}/{len(self.frames)}")

    def tecla_pressionada(self, event):
        tecla = event.char
        if tecla in TECLA_ESTADO:
            rotulo = TECLA_ESTADO[tecla]
            if rotulo != "IGNORAR":
                self.registrar_rotulo(rotulo)
            else:
                self.pular_frame()

    def registrar_rotulo(self, rotulo):
        caminho, estado_predito, nome = self.frames[self.index_atual]
        registro = [nome, estado_predito, rotulo]
        self.registros_novos.append(registro)
        print(f"Anotado: {nome} | Previsto: {estado_predito} | Real: {rotulo}")
        self.salvar_linha_csv(registro)
        self.index_atual += 1
        self.mostrar_proximo_frame()

    def pular_frame(self):
        print(f"Ignorado: {self.frames[self.index_atual][2]}")
        self.index_atual += 1
        self.mostrar_proximo_frame()

    def salvar_csv(self):
        if not self.registros_novos:
            return
        with open(OUTPUT_CSV, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.registros_novos)
        print(f"Novas anotações salvas em: {OUTPUT_CSV}")

    def mostrar_estatisticas(self):
        if not os.path.exists(OUTPUT_CSV):
            print("CSV não encontrado para estatísticas.")
            return

        y_true = []
        y_pred = []

        with open(OUTPUT_CSV, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                y_true.append(row["estado_real"])
                y_pred.append(row["estado_predito"])

        print("\n========= RELATÓRIO DE DESEMPENHO =========")
        print(classification_report(y_true, y_pred, labels=STATES_TO_VALIDAR, zero_division=0))

        acc = sum([1 for a, b in zip(y_true, y_pred) if a == b]) / len(y_true)
        print(f"Acurácia geral: {acc:.2%}")

        matrix = confusion_matrix(y_true, y_pred, labels=STATES_TO_VALIDAR)
        print("Matriz de confusão:")
        print(matrix)

        messagebox.showinfo("Resumo Estatístico",
                            f"Acurácia: {acc:.2%}\nTotal Frames: {len(y_true)}\n\nVeja o terminal para detalhes.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ValidadorGUI(root)
    root.mainloop()