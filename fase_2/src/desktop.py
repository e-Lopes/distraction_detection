"""Aplicação desktop Tk: consulta local e execução assíncrona da CLI canônica."""

from __future__ import annotations

import os
from pathlib import Path
from queue import Empty, Queue
import signal
import subprocess
import sys
from threading import Thread
import time

from .pipeline import _read_csv, build_plan, load_config
from .terminal import PROJECT_ROOT, PROTOCOL

STAGES = {"Comparar modelos": "screening", "Confirmar os escolhidos": "confirmation"}
METHODS = {"Todos": "all", "Medidas do rosto": "feature", "Semelhança entre trechos": "distance",
           "Padrões de movimento": "shapelet", "Transformações dos sinais": "transform",
           "Redes neurais": "deep"}
STATES = {"pending": "Aguardando", "running": "Em andamento", "completed": "Salvo",
          "reused": "Resultado anterior", "blocked": "Aguarda revisão",
          "dependency_missing": "Falta instalar componente", "failed": "Precisa de atenção"}


def simple_results(path):
    config = load_config(path)
    root = Path(config["outputs"]["root"])
    current = {r.run_id for r in build_plan(path, "all", "all", "all") if r.status == "completed"}
    current.update(Path(r.artifact).parent.name for r in build_plan(path, "all", "confirmation", "all")
                   if r.status == "completed" and Path(r.artifact).suffix == ".pt")
    paths = [root / "screening_metrics.csv", root / "confirmation_classical_metrics.csv",
             *(root / "temporal_metrics").rglob("*__runs.csv")]
    rows = [r for p in paths for r in _read_csv(p) if r.get("run_id") in current]
    lines = ["RESULTADOS DOS TREINAMENTOS ATUAIS", "",
             "A pontuação geral (F1) equilibra o reconhecimento dos três comportamentos.",
             "Quanto mais perto de 1, melhor. Ela não é a porcentagem de acertos.",
             "Também é preciso conferir fadiga e falsos alarmes antes de escolher um modelo.", ""]
    if not rows:
        lines.append("Ainda não há resultados de treinamentos concluídos com a configuração atual.")
    for row in rows[:100]:
        subset = "validação durante o desenvolvimento" if row.get("subset") == "validation" else "vídeo separado para teste"
        value = row.get("macro_f1_all_classes", "")
        lines.append(f"{row['model']} · divisão {row['fold']} · repetição {row.get('seed', '—')}\n"
                     f"  Avaliação: {subset}. Pontuação geral: {float(value):.3f}" if value else "Pontuação ainda indisponível.")
    if len(rows) > 100:
        lines.append("Mostrando os primeiros 100 resultados. A lista completa está nos arquivos abaixo.")
    lines.extend(["", f"Resultados completos: {root}",
                  f"Relatório científico e resultados anteriores: {config['outputs']['report']}"])
    return "\n".join(lines)


def simple_overview(path):
    config = load_config(path)
    series = Path(config["data"]["facial_series"])
    available = sum((series / f"{v}.csv").is_file() for v in config["data"]["videos"])
    runs = build_plan(path, "all", "screening", "all")
    done = sum(r.status in {"completed", "reused"} for r in runs)
    return ("SEU PRÓXIMO PASSO\n\n"
            + ("Coloque os quatro vídeos na pasta indicada e clique em Iniciar / continuar.\n"
               if available < len(config["data"]["videos"])
               else "Clique em Verificar dados ou Iniciar / continuar.\n")
            + "\nO programa vai:\n"
            "1. Ler os vídeos e salvar as medidas do rosto.\n"
            "2. Conferir os dados e separar o que será usado para aprender e avaliar.\n"
            "3. Treinar um modelo por vez e salvar os resultados antes de seguir.\n\n"
            f"Vídeos com medidas salvas: {available}/{len(config['data']['videos'])}\n"
            f"Treinamentos de comparação já salvos: {done}/{len(runs)}\n\n"
            "Você pode parar e continuar depois. Vídeos concluídos são reaproveitados; "
            "o vídeo interrompido é lido novamente. Redes neurais retomam da última rodada salva. "
            "Nos outros modelos, um ajuste interrompido é refeito; um ajuste já salvo é reaproveitado.\n\n"
            "Se faltar um arquivo ou componente, o programa mostrará o motivo. "
            "A confirmação dos modelos escolhidos depende de revisar os resultados da comparação.\n\n"
            f"Medidas salvas em: {series}\nResultados em: {config['outputs']['root']}")


class Job:
    """Um processo por vez; a thread leitora nunca acessa widgets Tk."""

    def __init__(self) -> None:
        self.events: Queue = Queue()
        self.process: subprocess.Popen | None = None
        self.reader: Thread | None = None

    @property
    def running(self) -> bool:
        return self.reader is not None and self.reader.is_alive()

    def start(self, command: list[str]) -> None:
        if self.running:
            raise RuntimeError("Já existe uma execução em andamento.")
        self.process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
            start_new_session=os.name == "posix",
        )
        self.reader = Thread(target=self._read, args=(self.process,), daemon=True)
        self.reader.start()

    def _read(self, process: subprocess.Popen) -> None:
        try:
            assert process.stdout is not None
            with process.stdout:
                for line in process.stdout:
                    self.events.put(("log", line))
        finally:
            self.events.put(("exit", process.wait()))

    def stop(self, *, force: bool = False) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
            elif force:
                process.kill()
            else:
                process.terminate()
        except ProcessLookupError:
            pass


class Desktop:
    def __init__(self, root, config_path: str | Path) -> None:
        import tkinter as tk
        from tkinter import ttk
        from tkinter.scrolledtext import ScrolledText

        self.root = root
        self.path = Path(config_path).resolve()
        self.job = Job()
        self.busy = False
        self.closing = False
        self.last_live_refresh = 0.0
        self.scope = tk.StringVar(value="Comparar modelos")
        self.paradigm = tk.StringVar(value="Todos")
        configured = load_config(self.path)["data"].get("video_dir", "")
        self.video_dir = tk.StringVar(value=str(configured))
        self.message = tk.StringVar(value="Pronto para consultar o protocolo.")
        root.title("Análise de atenção · Vídeos e treinamentos")
        root.geometry("1180x800")
        root.minsize(900, 600)
        root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(root)
        style.configure("Title.TLabel", font=("TkDefaultFont", 20, "bold"))
        style.configure("TButton", padding=(12, 7))

        header = ttk.Frame(root, padding=(20, 16))
        header.pack(fill="x")
        ttk.Label(header, text="Dos vídeos aos resultados", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="O programa usa a pasta configurada, confere os dados e salva cada treinamento.").pack(anchor="w")
        folder = ttk.Frame(root, padding=(20, 0, 20, 8))
        folder.pack(fill="x")
        ttk.Label(folder, text="1. Pasta dos vídeos:").pack(side="left")
        ttk.Label(folder, textvariable=self.video_dir, wraplength=700).pack(side="left", padx=12)

        controls = ttk.Frame(root, padding=(20, 0, 20, 12))
        controls.pack(fill="x")
        advanced = ttk.Frame(root, padding=(20, 0, 20, 12))
        show_advanced = tk.BooleanVar(value=False)
        ttk.Checkbutton(header, text="Mostrar opções avançadas", variable=show_advanced,
                        command=lambda: advanced.pack(fill="x", before=self.tabs) if show_advanced.get()
                        else advanced.pack_forget()).pack(anchor="w", pady=(8, 0))
        ttk.Label(advanced, text="Objetivo").pack(side="left")
        self.scope_box = ttk.Combobox(advanced, textvariable=self.scope, state="readonly",
                                     values=tuple(STAGES), width=24)
        self.scope_box.pack(side="left", padx=(6, 16))
        ttk.Label(advanced, text="Tipo de modelo").pack(side="left")
        self.paradigm_box = ttk.Combobox(
            advanced, textvariable=self.paradigm, state="readonly", width=28, values=tuple(METHODS))
        self.paradigm_box.pack(side="left", padx=6)
        ttk.Button(advanced, text="Protocolo técnico", command=lambda: self.show_document("protocol", PROTOCOL)).pack(side="left", padx=6)
        ttk.Button(advanced, text="Relatório técnico", command=lambda: self.show_document(
            "report", Path(load_config(self.path)["outputs"]["report"]))).pack(side="left")
        self.scope_box.bind("<<ComboboxSelected>>", lambda _: self.refresh_plan())
        self.paradigm_box.bind("<<ComboboxSelected>>", lambda _: self.refresh_plan())
        self.buttons = []
        for label, command in (("2. Verificar dados", lambda: self.execute("check-data")),
                               ("3. Iniciar / continuar", lambda: self.execute("chain")),
                               ("Atualizar tela", self.refresh),
                               ("Atualizar relatório", lambda: self.execute("report"))):
            button = ttk.Button(controls, text=label, command=command)
            button.pack(side="left", padx=3)
            self.buttons.append(button)

        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill="both", expand=True, padx=20)
        self.pages = {}
        self.texts = {}
        for key, title in (("overview", "Início"), ("plan", "Treinamentos"),
                           ("protocol", "Como funciona"), ("report", "Resultados"),
                           ("logs", "Acompanhamento")):
            page = ttk.Frame(self.tabs, padding=10)
            self.pages[key] = page
            self.tabs.add(page, text=title)
            if key != "plan":
                text = ScrolledText(page, wrap="word", padx=12, pady=12, borderwidth=0,
                                    font=("TkFixedFont", 11), state="disabled")
                text.pack(fill="both", expand=True)
                self.texts[key] = text
        plan_page = self.pages["plan"]
        ttk.Label(plan_page, text="Cada linha é um treinamento. O programa salva uma linha por vez, "
                  "antes de passar para a próxima.").pack(anchor="w", pady=(0, 10))
        columns = ("status", "model", "window", "fold", "seed", "representation", "reason")
        table_frame = ttk.Frame(plan_page)
        table_frame.pack(fill="both", expand=True)
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings")
        for column, label, width in zip(columns,
                ("Situação", "Modelo", "Trecho", "Divisão", "Repetição", "Entrada", "Motivo"),
                (130, 180, 65, 45, 65, 170, 400)):
            self.table.heading(column, text=label)
            self.table.column(column, width=width, minwidth=45, stretch=column == "reason")
        vertical = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        horizontal = ttk.Scrollbar(table_frame, orient="horizontal", command=self.table.xview)
        self.table.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.table.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        footer = ttk.Frame(root, padding=(20, 12))
        footer.pack(fill="x")
        self.stop_button = ttk.Button(footer, text="Parar", command=self.stop,
                                      state="disabled")
        self.stop_button.pack(side="right")
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=120)
        self.progress.pack(side="right", padx=12)
        ttk.Label(footer, textvariable=self.message, wraplength=700).pack(side="left")
        self.refresh()
        self.poll_id = root.after(100, self.poll)

    def set_text(self, key: str, value: str, *, append: bool = False) -> None:
        widget = self.texts[key]
        widget.configure(state="normal")
        if not append:
            widget.delete("1.0", "end")
        widget.insert("end", value)
        if key == "logs":
            # Mantém a janela responsiva em treinamentos longos.
            lines = int(widget.index("end-1c").split(".")[0])
            if lines > 10000:
                widget.delete("1.0", f"{lines - 10000}.0")
            widget.see("end")
        widget.configure(state="disabled")

    def refresh_plan(self, *, update_message: bool = True) -> None:
        try:
            plan = build_plan(self.path, "all", STAGES.get(self.scope.get(), self.scope.get()),
                              METHODS.get(self.paradigm.get(), self.paradigm.get()))
            self.table.delete(*self.table.get_children())
            for run in plan:
                reason = ("Precisa instalar o componente de séries temporais." if run.status == "dependency_missing"
                          else ("Cálculo pesado: " + run.reason.replace("pairs=", "comparações=")
                                .replace("limit=", "limite=").replace("cache=True", "resultado intermediário será salvo"))
                          if run.status == "blocked" and run.paradigm == "distance"
                          else "A confirmação aguarda a escolha dos melhores modelos da comparação."
                          if run.status == "blocked"
                          else "Resultado anterior disponível." if run.status == "reused" else run.reason)
                self.table.insert("", "end", values=(STATES.get(run.status, run.status), run.model,
                                  f"{run.window_size_frames} imagens", run.fold, run.seed,
                                  run.representation, reason))
            if update_message:
                completed = sum(run.status in {"completed", "reused"} for run in plan)
                running = sum(run.status == "running" for run in plan)
                text = f"{completed}/{len(plan)} treinamentos salvos."
                if running:
                    text += f" {running} em andamento; ele será salvo quando esta etapa terminar."
                self.message.set(text)
        except (OSError, ValueError, KeyError) as error:
            self.message.set(f"Falha ao consultar plano: {error}")

    def refresh(self) -> None:
        try:
            self.set_text("overview", simple_overview(self.path))
            self.set_text("protocol", "COMO FUNCIONA\n\n"
                "O programa mede a abertura dos olhos e da boca e os movimentos da cabeça.\n\n"
                "Os dados são divididos em pequenos trechos. Um modelo aprende com parte dos vídeos "
                "e é avaliado em outro vídeo que não participou do aprendizado.\n\n"
                "A comparação ajuda a escolher quais modelos merecem uma avaliação mais completa. "
                "Essa escolha precisa ser revisada antes da confirmação.\n\n"
                "O botão Parar preserva as etapas já concluídas. Clique em Iniciar / continuar "
                "para retomar o trabalho.\n\n"
                "Um resultado salvo não significa que o modelo já está pronto para uso real. "
                "Ainda é preciso conferir erros, falsos alarmes e reconhecimento de fadiga.\n\n"
                f"Protocolo científico completo: {PROTOCOL}")
            self.set_text("report", simple_results(self.path))
            self.refresh_plan()
        except (OSError, ValueError, KeyError) as error:
            self.message.set(f"Falha na consulta: {error}")

    def set_busy(self, busy: bool) -> None:
        self.busy = busy
        for button in self.buttons:
            button.configure(state="disabled" if busy else "normal")
        for box in (self.scope_box, self.paradigm_box):
            box.configure(state="disabled" if busy else "readonly")
        self.stop_button.configure(state="normal" if busy else "disabled")
        if busy:
            self.progress.start()
        else:
            self.progress.stop()

    def execute(self, action: str) -> None:
        if self.busy:
            return
        if action not in {"prepare", "train", "report", "check-data", "chain"}:
            raise ValueError(action)
        arguments = [action]
        if action in {"train", "chain"}:
            arguments.extend(["--scope", STAGES.get(self.scope.get(), self.scope.get()),
                              "--paradigm", METHODS.get(self.paradigm.get(), self.paradigm.get())])
        if action == "chain" and self.video_dir.get():
            arguments.extend(["--video-dir", self.video_dir.get()])
        command = [sys.executable, "-u", "-m", "fase_2", *arguments, "--config", str(self.path)]
        self.tabs.select(self.pages["logs"])
        self.set_text("logs", f"\nIniciando {action} · {self.scope.get()} / {self.paradigm.get()}\n", append=True)
        try:
            self.job.start(command)
        except (OSError, RuntimeError) as error:
            self.set_text("logs", f"Falha ao iniciar: {error}\n", append=True)
            self.message.set("Não foi possível iniciar a execução.")
            return
        self.set_busy(True)
        self.message.set("Trabalho em andamento. Veja a aba Acompanhamento.")

    def show_document(self, page, path):
        try:
            self.set_text(page, path.read_text(encoding="utf-8"))
            self.tabs.select(self.pages[page])
        except OSError:
            self.message.set("Documento ainda não disponível.")

    def stop(self) -> None:
        self.job.stop()
        self.message.set("Interrompendo execução…")
        process = self.job.process
        # O callback só pode encerrar o processo que motivou esta interrupção.
        self.root.after(3000, lambda: self.job.stop(force=True)
                        if self.job.process is process else None)

    def close(self) -> None:
        if self.busy:
            self.closing = True
            self.stop()
        else:
            self.root.after_cancel(self.poll_id)
            self.root.destroy()

    def poll(self) -> None:
        for _ in range(300):
            try:
                kind, value = self.job.events.get_nowait()
            except Empty:
                break
            if kind == "log":
                self.set_text("logs", value, append=True)
            else:
                self.set_text("logs", f"\nProcesso encerrado com código {value}. "
                              "Consulte o plano para runs pendentes ou bloqueados.\n", append=True)
                self.set_busy(False)
                self.refresh()
                self.message.set("Etapa finalizada. A lista de treinamentos foi atualizada."
                                 if value == 0 else "Há pendências. Veja o motivo na aba Acompanhamento.")
        if self.busy and time.monotonic() - self.last_live_refresh >= 3:
            self.refresh_plan(update_message=False)
            plan = build_plan(self.path, "all", STAGES.get(self.scope.get(), self.scope.get()),
                              METHODS.get(self.paradigm.get(), self.paradigm.get()))
            completed = sum(run.status in {"completed", "reused"} for run in plan)
            running = sum(run.status == "running" for run in plan)
            self.message.set(f"{completed}/{len(plan)} salvos"
                             + (f" · {running} em andamento" if running else ""))
            self.last_live_refresh = time.monotonic()
        if self.closing and not self.busy:
            self.root.destroy()
            return
        self.poll_id = self.root.after(100, self.poll)


def main(config_path: str | Path) -> int:
    try:
        import tkinter as tk
    except ImportError:
        print("Tkinter indisponível. Instale o suporte Tk do Python "
              "(pacote python3-tk no Debian/Ubuntu).", file=sys.stderr)
        return 2
    path = Path(config_path).resolve()
    if Path.cwd().resolve() != PROJECT_ROOT:
        print(f"Execute na raiz do repositório: {PROJECT_ROOT}", file=sys.stderr)
        return 2
    try:
        root = tk.Tk()
    except tk.TclError as error:
        print(f"Não foi possível abrir a janela desktop: {error}. "
              "Use uma sessão gráfica ou interface --summary para consulta textual.", file=sys.stderr)
        return 2
    Desktop(root, path)
    root.mainloop()
    return 0
