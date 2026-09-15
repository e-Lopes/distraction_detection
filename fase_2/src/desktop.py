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
METHODS = {"Todos": "all", "Regras fixas": "rule", "Medidas do rosto": "feature", "Semelhança entre trechos": "distance",
           "Padrões de movimento": "shapelet", "Transformações dos sinais": "transform",
           "Redes neurais": "deep"}
STATES = {"pending": "Pendente", "running": "Executando", "completed": "Concluída",
          "reused": "Reutilizada", "blocked": "Bloqueada",
          "dependency_missing": "Falta dependência", "failed": "Falhou"}
CONFIG_DIR = PROJECT_ROOT / "fase_2" / "configs"
EXPERIMENTS = {
    "Bateria binária completa": CONFIG_DIR / "binary_suite.yaml",
    "Binário · Comparação de modelos": CONFIG_DIR / "binary_experiment.yaml",
    "Referência histórica": CONFIG_DIR / "final_experiment.yaml",
    "Qualidade das medidas · 14/09": CONFIG_DIR / "measurement_experiment.yaml",
    "Famílias modernas · 14/09": CONFIG_DIR / "modern_experiment.yaml",
}

# All binary profiles are discoverable without editing YAML in the desktop.
import yaml
for _profile in yaml.safe_load((CONFIG_DIR / 'binary_suite.yaml').read_text())['suite']['profiles'][1:]:
    EXPERIMENTS['Binário · ' + _profile['label']] = PROJECT_ROOT / _profile['config']


def protocol_path(config):
    if config.get("target"):
        return PROJECT_ROOT / "fase_2/docs/protocols/binary_experiment_protocol.md"
    for key, filename in (("measurement_protocol", "measurement_quality_protocol.md"),
                          ("modern_protocol", "modern_families_protocol.md")):
        if config.get(key):
            return PROJECT_ROOT / "fase_2" / "docs" / "protocols" / filename
    return PROTOCOL


def execution_command(path, action, scope, paradigm, video_dir="", *, video="",
                      start_frame="0", max_frames="90", full=False, output="", allow_expensive=False):
    config = load_config(path)
    if action not in {"prepare", "train", "report", "check-data", "chain", "measurement-extract", "extract"}:
        raise ValueError(action)
    if config.get('suite') or (config.get('target') and action == 'chain'):
        args = [sys.executable, '-u', '-m', 'fase_2', 'suite', '--action', action,
                '--scope', STAGES.get(scope, scope), '--paradigm', METHODS.get(paradigm, paradigm),
                '--config', str(path)]
        if allow_expensive:
            args.append('--allow-expensive')
        return args
    # Os novos protocolos usam entradas próprias, sem extração legada implícita.
    if action == "chain" and (config.get("measurement_protocol") or config.get("modern_protocol")):
        action = "train"
    arguments = [action]
    if action in {"train", "chain"}:
        arguments.extend(["--scope", STAGES.get(scope, scope),
                          "--paradigm", METHODS.get(paradigm, paradigm)])
    if action in {"chain", "extract"} and video_dir:
        arguments.extend(["--video-dir", video_dir])
    if action == "measurement-extract":
        settings = config.get("measurement_protocol", {}).get("extraction", {})
        if video not in settings.get("videos", {}):
            raise ValueError("Selecione um vídeo do protocolo de qualidade das medidas.")
        start, count = int(start_frame), int(max_frames)
        if start < 0 or count <= 0 or (not full and count > settings["max_sample_frames"]):
            raise ValueError("Frame inicial deve ser não negativo e amostra deve respeitar o limite do protocolo.")
        if full and start != 0:
            raise ValueError("Extração completa deve começar no frame zero.")
        arguments.extend(["--video", video, "--start-frame", str(start), "--max-frames", str(count)])
        if full:
            arguments.append("--full")
        if output.strip():
            arguments.extend(["--output", output.strip()])
    if action == "train" and allow_expensive:
        arguments.append("--allow-expensive")
    return [sys.executable, "-u", "-m", "fase_2", *arguments, "--config", str(path)]


def result_rows(path, plan=None):
    """Only show metrics backed by completed, compatible runs."""
    config = load_config(path)
    if config.get('suite'):
        from .binary_suite import profiles
        return [dict(row, experiment=item['label']) for item in profiles(path)
                for row in result_rows(item['config'])]
    plan = build_plan(path, "all", "all", "all") if plan is None else plan
    current = {r.run_id for r in plan if r.status == "completed"}
    current.update(Path(r.artifact).parent.name for r in plan
                   if r.status == "completed" and Path(r.artifact).suffix == ".pt")
    root = Path(config["outputs"]["root"])
    paths = [root / "screening_metrics.csv", root / "confirmation_classical_metrics.csv",
             *(root / "temporal_metrics").rglob("*__runs.csv")]
    return [row for p in paths for row in _read_csv(p) if row.get("run_id") in current]


def simple_results(path):
    rows = result_rows(path)
    if not rows:
        return "Sem resultados compatíveis com a configuração atual."
    return f"{len(rows)} avaliações disponíveis."


def simple_overview(path):
    config = load_config(path)
    runs = build_plan(path, "all", "screening", "all")
    return f"{config['name']}\nTreinamentos previstos: {len(runs)}"


def plan_counts(plan):
    from collections import Counter
    counts = Counter(r.status for r in plan)
    return {"total": len(plan), "completed": counts['completed'], "reused": counts['reused'],
            "pending": counts['pending'], "running": counts['running'],
            "blocked": counts['blocked'] + counts['dependency_missing'] + counts['failed']}


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
    """Ações diárias em primeiro plano; configuração e documentação sob demanda."""

    def __init__(self, root, config_path):
        import tkinter as tk
        from tkinter import ttk
        from tkinter.scrolledtext import ScrolledText

        self.root = root
        self.path = Path(config_path).resolve()
        self.config = load_config(self.path)
        self.job = Job()
        self.busy = self.closing = self.stopping = False
        self.last_live_refresh = 0.0
        self.snapshot_events = Queue()
        self.snapshot_thread = None
        self.refresh_pending = False
        self.snapshot_key = None
        self.plan = []
        self.last_outcome = ''
        self.rows_by_id = {}
        self.options_window = None
        self.scope = tk.StringVar(value="Comparar modelos")
        self.paradigm = tk.StringVar(value="Todos")
        self.experiment = tk.StringVar(value=self.experiment_name())
        self.video_dir = tk.StringVar(value=str(self.config['data'].get('video_dir', '')))
        self.video = tk.StringVar(value="")
        self.start_frame = tk.StringVar(value="0")
        self.max_frames = tk.StringVar(value="90")
        self.extraction_output = tk.StringVar(value="")
        self.full = tk.BooleanVar(value=False)
        self.allow_expensive = tk.BooleanVar(value=False)
        self.message = tk.StringVar(value="")
        self.detail = tk.StringVar(value="Selecione uma execução para ver os detalhes.")
        self.selection_label = tk.StringVar()
        self.result_hint = tk.StringVar(value="")
        self.counter_vars = {key: tk.StringVar(value="—") for key in
                             ('total', 'completed', 'reused', 'pending', 'blocked')}
        root.title("Experimentos · Atenção e Distração")
        root.geometry("1080x700")
        root.minsize(820, 540)
        root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(root)
        style.configure("Title.TLabel", font=("TkDefaultFont", 18, "bold"))
        style.configure("Value.TLabel", font=("TkDefaultFont", 19, "bold"))
        style.configure("TButton", padding=(10, 6))
        style.configure("Treeview", rowheight=29)
        style.configure("Treeview.Heading", font=("TkDefaultFont", 10, "bold"))

        header = ttk.Frame(root, padding=(20, 16, 20, 10))
        header.pack(fill="x")
        ttk.Label(header, text="Experimentos", style="Title.TLabel").pack(side="left")
        self.options_button = ttk.Button(header, text="Opções…", command=self.open_options)
        self.options_button.pack(side="right")
        self.experiment_box = ttk.Combobox(header, textvariable=self.experiment,
            values=tuple(EXPERIMENTS), state="readonly", width=35)
        self.experiment_box.pack(side="right", padx=12)
        self.experiment_box.bind('<<ComboboxSelected>>',
            lambda _: self.select_config(EXPERIMENTS[self.experiment.get()]))

        actions = ttk.Frame(root, padding=(20, 0, 20, 12))
        actions.pack(fill="x")
        self.buttons = []
        for label, action in (("Verificar dados", "check-data"), ("Iniciar / continuar", "chain")):
            button = ttk.Button(actions, text=label, command=lambda a=action: self.execute(a))
            button.pack(side="left", padx=(0, 8))
            self.buttons.append(button)
        self.new_round_button = ttk.Button(actions, text="Nova rodada binária (do zero)", command=self.create_round)
        self.new_round_button.pack(side='left', padx=(0,8))
        self.buttons.append(self.new_round_button)
        self.stop_button = ttk.Button(actions, text="Parar", command=self.stop, state="disabled")
        self.stop_button.pack(side="left")
        self.refresh_button = ttk.Button(actions, text="Atualizar", command=self.refresh)
        self.refresh_button.pack(side="right")

        stats = ttk.Frame(root, padding=(20, 0, 20, 14))
        stats.pack(fill="x")
        for index, (key, label) in enumerate((('total', 'Execuções'), ('completed', 'Concluídas'),
                ('reused', 'Reutilizadas'), ('pending', 'Pendentes'), ('blocked', 'Impedimentos'))):
            card = ttk.Frame(stats, padding=(12, 8), relief="groove")
            card.grid(row=0, column=index, sticky="ew", padx=(0, 8 if index < 4 else 0))
            stats.columnconfigure(index, weight=1)
            ttk.Label(card, textvariable=self.counter_vars[key], style="Value.TLabel").pack(anchor="w")
            ttk.Label(card, text=label).pack(anchor="w")

        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill="both", expand=True, padx=20)
        self.pages = {}
        self.texts = {}
        for key, label in (("plan", "Experimentos"), ("report", "Resultados"), ("logs", "Logs")):
            page = ttk.Frame(self.tabs, padding=12)
            self.pages[key] = page
            self.tabs.add(page, text=label)

        ttk.Label(self.pages['plan'], textvariable=self.selection_label).pack(anchor='w', pady=(0, 8))
        self.table = self.make_table(self.pages['plan'],
            ('status', 'experiment', 'model', 'window', 'fold', 'seed', 'representation'),
            ('Situação', 'Experimento', 'Modelo', 'Janela (frames)', 'Fold', 'Seed', 'Entrada'),
            (130, 190, 180, 100, 55, 55, 100))
        self.table.bind('<<TreeviewSelect>>', self.select_run)
        self.confirm_button = ttk.Button(self.pages['plan'], text='Confirmar candidato selecionado',
                                         command=self.confirm_candidate)
        self.confirm_button.pack(anchor='w',pady=(8,0))
        self.buttons.append(self.confirm_button)
        ttk.Label(self.pages['plan'], textvariable=self.detail, wraplength=920).pack(
            fill='x', pady=(10, 0))

        results_bar = ttk.Frame(self.pages['report'])
        results_bar.pack(fill='x', pady=(0, 10))
        self.report_button = ttk.Button(results_bar, text="Gerar relatório", command=lambda: self.execute('report'))
        self.report_button.pack(side='left', padx=(0, 8))
        self.buttons.append(self.report_button)
        ttk.Button(results_bar, text="Abrir relatório", command=lambda: self.show_document(
            'Relatório', Path(self.config['outputs']['report']))).pack(side='left', padx=(0, 8))
        ttk.Button(results_bar, text="Gráficos", command=self.open_gallery).pack(side='left')
        ttk.Label(self.pages['report'], textvariable=self.result_hint, wraplength=900).pack(anchor='w', pady=(0, 8))
        self.results_table = self.make_table(self.pages['report'],
            ('experiment', 'model', 'representation', 'window', 'fold', 'seed', 'subset', 'f1'),
            ('Experimento', 'Modelo', 'Entrada', 'Janela', 'Fold', 'Seed', 'Avaliação', 'Macro F1'),
            (180, 160, 100, 65, 55, 55, 100, 80))
        log = ScrolledText(self.pages['logs'], wrap='word', state='disabled',
                           font=('TkFixedFont', 10), borderwidth=0, padx=8, pady=8)
        log.pack(fill='both', expand=True)
        self.texts['logs'] = log

        footer = ttk.Frame(root, padding=(20, 10))
        footer.pack(fill='x')
        self.progress = ttk.Progressbar(footer, mode='indeterminate', length=100)
        ttk.Label(footer, textvariable=self.message, wraplength=850).pack(side='left')
        self.refresh()
        self.poll_id = root.after(100, self.poll)

    def experiment_name(self):
        return next((name for name, path in EXPERIMENTS.items() if path.resolve() == self.path),
                    self.path.stem)

    def make_table(self, parent, columns, labels, widths):
        from tkinter import ttk
        frame = ttk.Frame(parent)
        frame.pack(fill='both', expand=True)
        table = ttk.Treeview(frame, columns=columns, show='headings', selectmode='browse')
        for key, label, width in zip(columns, labels, widths):
            table.heading(key, text=label)
            table.column(key, width=width, minwidth=55, stretch=key in {'model', 'representation'})
        vertical = ttk.Scrollbar(frame, orient='vertical', command=table.yview)
        horizontal = ttk.Scrollbar(frame, orient='horizontal', command=table.xview)
        table.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        table.grid(row=0, column=0, sticky='nsew')
        vertical.grid(row=0, column=1, sticky='ns')
        horizontal.grid(row=1, column=0, sticky='ew')
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        return table

    def select_config(self, path):
        if self.busy:
            return
        try:
            config = load_config(path)
        except Exception as error:
            self.message.set(f"Configuração inválida: {error}")
            self.experiment.set(self.experiment_name())
            return
        self.path, self.config = Path(path).resolve(), config
        self.last_outcome = ''
        self.experiment.set(self.experiment_name())
        self.scope.set('Comparar modelos')
        self.paradigm.set('Todos')
        self.video_dir.set(str(config['data'].get('video_dir', '')))
        self.full.set(False)
        self.start_frame.set('0')
        self.max_frames.set('90')
        self.video.set('')
        self.extraction_output.set('')
        if self.options_window is not None:
            self.options_window.destroy()
            self.options_window = None
        self.refresh()

    def confirm_candidate(self):
        if self.busy:
            return
        selection=self.table.selection()
        if not selection:
            self.message.set('Selecione um candidato na tabela de experimentos.')
            return
        from .binary_suite import create_confirmation
        try:
            path=create_confirmation(self.path,selection[0])
            name=load_config(path)['name']
            EXPERIMENTS[name]=path.resolve()
            self.experiment_box.configure(values=tuple(EXPERIMENTS))
            self.select_config(path)
            self.scope.set('Confirmar os escolhidos')
            self.refresh()
            self.message.set('Candidato congelado pela validação. Iniciar / continuar executará a confirmação; redes usam cinco seeds.')
        except Exception as error:
            self.message.set(f'Não foi possível confirmar: {error}')

    def create_round(self):
        if self.busy:
            return
        if not self.config.get('target'):
            self.message.set('Selecione a bateria binária para criar uma nova rodada.')
            return
        from .binary_suite import new_round
        try:
            path = new_round(self.path)
            name = load_config(path)['name']
            EXPERIMENTS[name] = path.resolve()
            self.experiment_box.configure(values=tuple(EXPERIMENTS))
            self.select_config(path)
            self.message.set('Nova rodada vazia criada. Clique em Iniciar / continuar para treinar do zero.')
        except Exception as error:
            self.message.set(f'Não foi possível criar a rodada: {error}')

    def choose_config(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(parent=self.root, initialdir=CONFIG_DIR,
            filetypes=[('Configuração YAML', '*.yaml *.yml')])
        if path:
            self.select_config(path)

    def open_options(self):
        import tkinter as tk
        from tkinter import ttk
        if self.busy:
            return
        if self.options_window is not None and self.options_window.winfo_exists():
            self.options_window.lift()
            return
        window = tk.Toplevel(self.root)
        self.options_window = window
        window.title('Opções do experimento')
        window.transient(self.root)
        window.resizable(True, True)
        body = ttk.Frame(window, padding=18)
        body.pack(fill='both', expand=True)
        body.columnconfigure(1, weight=1)
        ttk.Label(body, text='Configuração').grid(row=0, column=0, sticky='w', padx=(0, 12))
        ttk.Label(body, text=self.path.name).grid(row=0, column=1, sticky='w')
        ttk.Button(body, text='Abrir YAML…', command=self.choose_config).grid(row=0, column=2)
        for row, label, variable, values in ((1, 'Etapa', self.scope, tuple(STAGES)),
                (2, 'Família', self.paradigm, tuple(METHODS))):
            ttk.Label(body, text=label).grid(row=row, column=0, sticky='w')
            box = ttk.Combobox(body, textvariable=variable, values=values, state='readonly', width=32)
            box.grid(row=row, column=1, columnspan=2, sticky='ew', pady=5)
            box.bind('<<ComboboxSelected>>', lambda _: self.refresh())
        paths = ttk.LabelFrame(body, text='Arquivos', padding=10)
        paths.grid(row=3, column=0, columnspan=3, sticky='ew', pady=10)
        for label, value in (('Entradas', self.config['data']['facial_series']),
                             ('Saídas', self.config['outputs']['root']),
                             ('Python', sys.executable)):
            ttk.Label(paths, text=f'{label}: {value}', wraplength=620).pack(anchor='w', pady=2)
        tools = ttk.Frame(body)
        tools.grid(row=4, column=0, columnspan=3, sticky='w')
        ttk.Button(tools, text='Preparar entradas', command=lambda: self.execute('prepare')).pack(side='left', padx=(0, 8))
        ttk.Button(tools, text='Protocolo', command=lambda: self.show_document(
            'Protocolo', protocol_path(self.config))).pack(side='left')

        if self.config.get('target') and not self.config.get('suite') and not self.config.get('measurement_protocol'):
            ttk.Label(body, text='Pasta dos vídeos').grid(row=5,column=0,sticky='w')
            ttk.Entry(body,textvariable=self.video_dir).grid(row=5,column=1,sticky='ew')
            ttk.Button(body,text='Extrair indicadores',command=lambda:self.execute('extract')).grid(row=5,column=2)

        if self.config.get('measurement_protocol'):
            panel = ttk.LabelFrame(body, text='Extração de medidas', padding=12)
            panel.grid(row=5, column=0, columnspan=3, sticky='ew', pady=(14, 0))
            panel.columnconfigure(1, weight=1)
            videos = tuple(self.config['measurement_protocol']['extraction']['videos'])
            if self.video.get() not in videos:
                self.video.set(videos[0] if videos else '')
            ttk.Label(panel, text='Vídeo').grid(row=0, column=0, sticky='w')
            ttk.Combobox(panel, textvariable=self.video, values=videos, state='readonly').grid(row=0, column=1, sticky='ew')
            for row, label, variable in ((1, 'Frame inicial', self.start_frame),
                    (2, 'Frames da amostra', self.max_frames), (3, 'Pasta de saída (opcional)', self.extraction_output)):
                ttk.Label(panel, text=label).grid(row=row, column=0, sticky='w', padx=(0, 12))
                ttk.Entry(panel, textvariable=variable).grid(row=row, column=1, sticky='ew', pady=4)
            ttk.Checkbutton(panel, text='Vídeo completo — exige âncora verificada', variable=self.full).grid(
                row=4, column=0, columnspan=2, sticky='w', pady=6)
            ttk.Button(panel, text='Extrair', command=lambda: self.execute('measurement-extract')).grid(
                row=5, column=1, sticky='e')
        ttk.Checkbutton(body, text='Executar DTW mesmo acima do orçamento de pares (pode demorar horas)',
                        variable=self.allow_expensive).grid(row=7, column=0, columnspan=3, sticky='w', pady=8)
        ttk.Button(body, text='Fechar', command=window.destroy).grid(row=6, column=2, sticky='e', pady=(12, 0))
        window.bind('<Destroy>', lambda event: setattr(self, 'options_window', None)
                    if event.widget is window else None)

    def refresh(self):
        key = (str(self.path), STAGES.get(self.scope.get(), self.scope.get()),
               METHODS.get(self.paradigm.get(), self.paradigm.get()))
        if key != self.snapshot_key:
            self.table.delete(*self.table.get_children())
            self.results_table.delete(*self.results_table.get_children())
            self.rows_by_id.clear()
            self.plan = []
            self.detail.set('Selecione uma execução para ver os detalhes.')
            for var in self.counter_vars.values():
                var.set('—')
        self.snapshot_key = key
        self.selection_label.set(('Atenção = alert · Distração = fatigue + distraction · ' if self.config.get('target') else 'Três classes · ') + f'{self.scope.get()} · {self.paradigm.get()}')
        if self.snapshot_thread is not None and self.snapshot_thread.is_alive():
            self.refresh_pending = True
            return
        def read_snapshot():
            try:
                plan = build_plan(key[0], 'all', key[1], key[2])
                rows = result_rows(key[0], plan)
                self.snapshot_events.put((key, plan, rows, None))
            except Exception as error:
                self.snapshot_events.put((key, [], [], str(error)))
        self.snapshot_thread = Thread(target=read_snapshot, daemon=True)
        self.snapshot_thread.start()
        self.last_live_refresh = time.monotonic()

    def refresh_plan(self, *, update_message=True):
        self.refresh()

    def apply_snapshot(self, key, plan, rows, error):
        current = (str(self.path), STAGES.get(self.scope.get(), self.scope.get()),
                   METHODS.get(self.paradigm.get(), self.paradigm.get()))
        if key != current:
            self.refresh_pending = True
            return
        if error:
            self.message.set(f'Falha ao consultar: {error}')
            return
        self.plan = plan
        selection = self.table.selection()
        self.table.delete(*self.table.get_children())
        self.rows_by_id = {r.run_id: r for r in plan}
        for run in plan:
            self.table.insert('', 'end', iid=run.run_id, values=(STATES.get(run.status, run.status),
                run.run_id.split('::')[0] if '::' in run.run_id else self.config['name'],
                run.model, run.window_size_frames, run.fold, run.seed, run.representation))
        if selection and selection[0] in self.rows_by_id:
            self.table.selection_set(selection[0])
        for name, value in plan_counts(plan).items():
            if name in self.counter_vars:
                self.counter_vars[name].set(str(value))
        self.results_table.delete(*self.results_table.get_children())
        for row in rows:
            value = row.get('macro_f1_all_classes', '')
            try:
                score = f'{float(value):.3f}' if value else '—'
            except (ValueError, TypeError):
                score = '—'
            subset = {'validation': 'Validação', 'test': 'Teste'}.get(row.get('subset'), row.get('subset', '—'))
            self.results_table.insert('', 'end', values=(row.get('experiment',self.config['name']), row.get('model', '—'),
                row.get('representation', '—'), row.get('window_size_frames', '—'),
                row.get('fold', '—'), row.get('seed', '—'), subset, score))
        self.result_hint.set(f'{len(rows)} avaliações compatíveis com a seleção.' if rows else
                             'Sem resultados compatíveis nesta seleção. Histórico disponível no relatório e nos gráficos.')
        if not self.busy:
            self.message.set(self.last_outcome or ('' if plan else 'Nenhuma execução nesta seleção.'))

    def select_run(self, event=None):
        selected = self.table.selection()
        if selected and selected[0] in self.rows_by_id:
            run = self.rows_by_id[selected[0]]
            detail = run.reason or ('Resultado disponível.' if run.status in {'completed', 'reused'}
                                    else 'Aguardando execução.')
            self.detail.set(f'{run.model} · Fold {run.fold} · Seed {run.seed}: {detail}')

    def set_text(self, key, value, *, append=False):
        widget = self.texts[key]
        widget.configure(state='normal')
        if not append:
            widget.delete('1.0', 'end')
        widget.insert('end', value)
        lines = int(widget.index('end-1c').split('.')[0])
        if lines > 10000:
            widget.delete('1.0', f'{lines - 10000}.0')
        widget.see('end')
        widget.configure(state='disabled')

    def set_busy(self, busy):
        self.busy = busy
        for button in self.buttons + [self.options_button]:
            button.configure(state='disabled' if busy else 'normal')
        self.experiment_box.configure(state='disabled' if busy else 'readonly')
        self.stop_button.configure(state='normal' if busy else 'disabled')
        if busy:
            self.progress.pack(side='right', padx=(10, 0))
            self.progress.start()
            if self.options_window is not None:
                self.options_window.destroy()
        else:
            self.progress.stop()
            self.progress.pack_forget()

    def execute(self, action):
        if self.busy:
            return
        try:
            command = execution_command(self.path, action, self.scope.get(), self.paradigm.get(),
                self.video_dir.get(), video=self.video.get(), start_frame=self.start_frame.get(),
                max_frames=self.max_frames.get(), full=self.full.get(), output=self.extraction_output.get(),
                allow_expensive=self.allow_expensive.get())
            self.job.start(command)
        except Exception as error:
            self.message.set(f'Não foi possível iniciar: {error}')
            return
        self.stopping = False
        self.last_outcome = ''
        self.tabs.select(self.pages['logs'])
        self.set_text('logs', f'\n{command[4]} · {self.experiment.get()} · {self.scope.get()} / {self.paradigm.get()}\n', append=True)
        self.set_busy(True)
        self.message.set('Em execução…')

    def show_document(self, title, path):
        import tkinter as tk
        from tkinter.scrolledtext import ScrolledText
        try:
            content = path.read_text(encoding='utf-8')
        except OSError:
            self.message.set('Documento ainda não disponível.')
            return
        window = tk.Toplevel(self.root)
        window.title(title)
        window.geometry('850x620')
        text = ScrolledText(window, wrap='word', padx=16, pady=16)
        text.pack(fill='both', expand=True)
        text.insert('end', content)
        text.configure(state='disabled')

    def open_gallery(self):
        import webbrowser
        from ..scripts.index_results import generate_index
        try:
            target = generate_index(PROJECT_ROOT / 'fase_2/results')
            if not webbrowser.open(target.as_uri()):
                self.message.set(f'Galeria: {target}')
        except OSError as error:
            self.message.set(f'Não foi possível abrir gráficos: {error}')

    def stop(self):
        self.stopping = True
        self.job.stop()
        self.message.set('Interrompendo…')
        process = self.job.process
        self.root.after(3000, lambda: self.job.stop(force=True) if self.job.process is process else None)

    def close(self):
        if self.busy:
            self.closing = True
            self.stop()
        else:
            self.root.after_cancel(self.poll_id)
            self.root.destroy()

    def poll(self):
        for _ in range(300):
            try:
                kind, value = self.job.events.get_nowait()
            except Empty:
                break
            if kind == 'log':
                self.set_text('logs', value, append=True)
            else:
                self.set_text('logs', f'\nProcesso encerrado: código {value}.\n', append=True)
                self.set_busy(False)
                self.refresh()
                self.last_outcome = ('Execução interrompida.' if self.stopping else
                    'Comando finalizado. Consulte o plano.' if value == 0 else 'Falha na execução. Consulte os logs.')
                self.message.set(self.last_outcome)
        while True:
            try:
                snapshot = self.snapshot_events.get_nowait()
            except Empty:
                break
            self.apply_snapshot(*snapshot)
        if self.closing and not self.busy:
            self.root.destroy()
            return
        if self.refresh_pending and (self.snapshot_thread is None or not self.snapshot_thread.is_alive()):
            self.refresh_pending = False
            self.refresh()
        elif self.busy and time.monotonic() - self.last_live_refresh >= 5:
            self.refresh()
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
