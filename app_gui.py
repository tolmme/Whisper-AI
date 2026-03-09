import contextlib
import os
import queue
import subprocess
import sys
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import main as core

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    DND_FILES = None
    TkinterDnD = None


LOCAL_MODEL_OPTIONS = [
    ("small.en", "ENG Small (`small.en`, ~2 GB RAM)"),
    ("medium.en", "ENG Medium (`medium.en`, ~5 GB RAM)"),
    ("small", "Multilingual Small (`small`, ~2 GB RAM)"),
    ("medium", "Multilingual Medium (`medium`, ~5 GB RAM)"),
    ("large-v3", "Multilingual Large-v3 (`large-v3`, ~9-10 GB RAM)"),
]

OPENAI_MODEL_OPTIONS = [
    ("gpt-4o-transcribe-diarize", "gpt-4o-transcribe-diarize (speakers + timestamps)"),
    ("gpt-4o-transcribe", "gpt-4o-transcribe"),
    ("gpt-4o-mini-transcribe", "gpt-4o-mini-transcribe"),
]

AUDIO_FILETYPES = [
    ("Audio/Video files", "*.mp3 *.wav *.m4a *.mp4 *.aac"),
    ("All files", "*.*"),
]

COLOR_BG = "#E8EDF5"
COLOR_PANEL = "#FFFFFF"
COLOR_PANEL_ALT = "#F6F8FC"
COLOR_BORDER = "#D5DDEC"
COLOR_TEXT = "#101828"
COLOR_MUTED = "#5F6C83"
COLOR_ACCENT = "#1F6BFF"
COLOR_ACCENT_ACTIVE = "#0F58E8"
COLOR_DANGER = "#D94242"
COLOR_DANGER_ACTIVE = "#BD2C2C"
COLOR_INPUT_BG = "#FFFFFF"

COLOR_SIDEBAR = "#0F1A2E"
COLOR_SIDEBAR_ALT = "#15233C"
COLOR_SIDEBAR_TEXT = "#E8EEFA"
COLOR_SIDEBAR_MUTED = "#A6B3CD"
COLOR_LOG_BG = "#0B1220"
COLOR_LOG_TEXT = "#D8E0EF"
COLOR_DROP_BG = "#EEF4FF"
COLOR_DROP_BORDER = "#B9CCF6"
COLOR_DROP_ACTIVE_BG = "#DDEBFF"
COLOR_DROP_ACTIVE_BORDER = "#1F6BFF"

FONT_PRIMARY = "Avenir Next" if sys.platform == "darwin" else "Segoe UI Variable"
FONT_MONO = "SF Mono" if sys.platform == "darwin" else "Consolas"
TEXT_SM = 12
TEXT_BASE = 13


class QueueLogWriter:
    def __init__(self, output_queue):
        self.output_queue = output_queue

    def write(self, text):
        if text:
            self.output_queue.put(text)

    def flush(self):
        return None


class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Transcriber")
        self.root.geometry("1280x840")
        self.root.minsize(1180, 760)
        self.root.configure(bg=COLOR_BG)

        self.file_paths = []
        self.item_to_path = {}
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.log_queue = queue.Queue()

        self.source_var = tk.StringVar(value="local")
        self.timestamps_var = tk.BooleanVar(value=True)
        self.current_file_var = tk.StringVar(value="Current file: -")
        self.overall_var = tk.StringVar(value="Queue progress: 0/0")
        self.model_display_var = tk.StringVar()
        self.sidebar_queue_var = tk.StringVar(value="0 files in queue")
        self.sidebar_done_var = tk.StringVar(value="0 completed")
        self.sidebar_backend_var = tk.StringVar(value="Local backend")
        self.sidebar_timestamps_var = tk.StringVar(value="Timestamps: On")
        self.drop_hint_var = tk.StringVar(value="Drop audio files or folders here")

        self.model_display_to_value = {}
        self.drag_and_drop_enabled = bool(TkinterDnD and DND_FILES)

        self._apply_theme()
        self._build_ui()
        self._set_model_options()
        self._setup_drag_and_drop()
        self._poll_log_queue()

    def _apply_theme(self):
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        self.style = style
        default_font = (FONT_PRIMARY, TEXT_BASE)

        style.configure(
            ".",
            background=COLOR_BG,
            foreground=COLOR_TEXT,
            font=default_font,
            borderwidth=0,
            relief="flat",
        )

        style.configure("App.TFrame", background=COLOR_BG)
        style.configure("Sidebar.TFrame", background=COLOR_SIDEBAR)
        style.configure("SidebarCard.TFrame", background=COLOR_SIDEBAR_ALT, bordercolor="#25385F", borderwidth=1, relief="solid")
        style.configure("TopCard.TFrame", background=COLOR_PANEL, bordercolor=COLOR_BORDER, borderwidth=1, relief="solid")
        style.configure("Card.TLabelframe", background=COLOR_PANEL, bordercolor=COLOR_BORDER, borderwidth=1, relief="solid")
        style.configure("Card.TLabelframe.Label", background=COLOR_PANEL, foreground=COLOR_MUTED, font=(FONT_PRIMARY, TEXT_SM, "bold"))
        style.configure("Toolbar.TFrame", background=COLOR_BG)
        style.configure("TLabel", background=COLOR_PANEL, foreground=COLOR_TEXT, font=(FONT_PRIMARY, TEXT_BASE))

        style.configure("SidebarTitle.TLabel", background=COLOR_SIDEBAR, foreground=COLOR_SIDEBAR_TEXT, font=(FONT_PRIMARY, 23, "bold"))
        style.configure("SidebarSubtitle.TLabel", background=COLOR_SIDEBAR, foreground=COLOR_SIDEBAR_MUTED, font=(FONT_PRIMARY, TEXT_SM))
        style.configure("SidebarMetaLabel.TLabel", background=COLOR_SIDEBAR_ALT, foreground=COLOR_SIDEBAR_MUTED, font=(FONT_PRIMARY, TEXT_SM))
        style.configure("SidebarMetaValue.TLabel", background=COLOR_SIDEBAR_ALT, foreground=COLOR_SIDEBAR_TEXT, font=(FONT_PRIMARY, 14, "bold"))
        style.configure("SidebarFoot.TLabel", background=COLOR_SIDEBAR, foreground=COLOR_SIDEBAR_MUTED, font=(FONT_PRIMARY, TEXT_SM))

        style.configure("TopTitle.TLabel", background=COLOR_PANEL, foreground=COLOR_TEXT, font=(FONT_PRIMARY, 20, "bold"))
        style.configure("TopSubtitle.TLabel", background=COLOR_PANEL, foreground=COLOR_MUTED, font=(FONT_PRIMARY, TEXT_SM))
        style.configure("Muted.TLabel", background=COLOR_PANEL, foreground=COLOR_MUTED, font=(FONT_PRIMARY, TEXT_SM))
        style.configure("TopMuted.TLabel", background=COLOR_PANEL, foreground=COLOR_MUTED, font=(FONT_PRIMARY, TEXT_SM, "bold"))
        style.configure(
            "DropHint.TLabel",
            background=COLOR_DROP_BG,
            foreground=COLOR_MUTED,
            bordercolor=COLOR_DROP_BORDER,
            lightcolor=COLOR_DROP_BORDER,
            darkcolor=COLOR_DROP_BORDER,
            relief="solid",
            borderwidth=1,
            padding=(12, 12),
            font=(FONT_PRIMARY, TEXT_SM, "bold"),
        )
        style.configure(
            "DropHintActive.TLabel",
            background=COLOR_DROP_ACTIVE_BG,
            foreground=COLOR_ACCENT,
            bordercolor=COLOR_DROP_ACTIVE_BORDER,
            lightcolor=COLOR_DROP_ACTIVE_BORDER,
            darkcolor=COLOR_DROP_ACTIVE_BORDER,
            relief="solid",
            borderwidth=1,
            padding=(12, 12),
            font=(FONT_PRIMARY, TEXT_SM, "bold"),
        )

        style.configure("TRadiobutton", background=COLOR_PANEL, foreground=COLOR_TEXT, font=(FONT_PRIMARY, TEXT_SM))
        style.map(
            "TRadiobutton",
            background=[("active", COLOR_PANEL)],
            foreground=[("disabled", "#8C9AB2")],
        )
        style.configure("TCheckbutton", background=COLOR_PANEL, foreground=COLOR_TEXT, font=(FONT_PRIMARY, TEXT_SM))
        style.map("TCheckbutton", foreground=[("disabled", "#8C9AB2")], background=[("active", COLOR_PANEL)])

        style.configure(
            "TCombobox",
            fieldbackground=COLOR_INPUT_BG,
            background=COLOR_INPUT_BG,
            foreground=COLOR_TEXT,
            arrowcolor=COLOR_MUTED,
            bordercolor=COLOR_BORDER,
            lightcolor=COLOR_BORDER,
            darkcolor=COLOR_BORDER,
            insertcolor=COLOR_TEXT,
            padding=7,
            font=(FONT_PRIMARY, TEXT_BASE),
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", COLOR_INPUT_BG)],
            foreground=[("readonly", COLOR_TEXT), ("disabled", "#8C9AB2")],
        )

        style.configure(
            "Secondary.TButton",
            background=COLOR_PANEL_ALT,
            foreground=COLOR_TEXT,
            bordercolor=COLOR_BORDER,
            lightcolor=COLOR_BORDER,
            darkcolor=COLOR_BORDER,
            padding=(12, 9),
            font=(FONT_PRIMARY, TEXT_SM, "bold"),
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#E9EEF8"), ("disabled", "#F3F6FB")],
            foreground=[("disabled", "#9CA3AF")],
        )

        style.configure(
            "Primary.TButton",
            background=COLOR_ACCENT,
            foreground="#FFFFFF",
            bordercolor=COLOR_ACCENT,
            lightcolor=COLOR_ACCENT,
            darkcolor=COLOR_ACCENT,
            padding=(12, 9),
            font=(FONT_PRIMARY, TEXT_SM, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[("active", COLOR_ACCENT_ACTIVE), ("disabled", "#9FC0FF")],
            foreground=[("disabled", "#E5E7EB")],
        )

        style.configure(
            "Danger.TButton",
            background=COLOR_DANGER,
            foreground="#FFFFFF",
            bordercolor=COLOR_DANGER,
            lightcolor=COLOR_DANGER,
            darkcolor=COLOR_DANGER,
            padding=(12, 9),
            font=(FONT_PRIMARY, TEXT_SM, "bold"),
        )
        style.map(
            "Danger.TButton",
            background=[("active", COLOR_DANGER_ACTIVE), ("disabled", "#FCA5A5")],
            foreground=[("disabled", "#FEF2F2")],
        )

        style.configure(
            "Horizontal.TProgressbar",
            troughcolor="#DEE6F2",
            background=COLOR_ACCENT,
            bordercolor=COLOR_BORDER,
            lightcolor=COLOR_ACCENT,
            darkcolor=COLOR_ACCENT,
            thickness=8,
        )

        style.configure(
            "Treeview",
            background=COLOR_PANEL,
            fieldbackground=COLOR_PANEL,
            foreground=COLOR_TEXT,
            rowheight=30,
            bordercolor=COLOR_BORDER,
            lightcolor=COLOR_BORDER,
            darkcolor=COLOR_BORDER,
            font=(FONT_PRIMARY, TEXT_SM),
        )
        style.map("Treeview", background=[("selected", "#DBEAFE")], foreground=[("selected", COLOR_TEXT)])
        style.configure(
            "Treeview.Heading",
            background=COLOR_PANEL_ALT,
            foreground=COLOR_MUTED,
            bordercolor=COLOR_BORDER,
            lightcolor=COLOR_BORDER,
            darkcolor=COLOR_BORDER,
            relief="flat",
            font=(FONT_PRIMARY, TEXT_SM, "bold"),
        )
        style.map("Treeview.Heading", background=[("active", COLOR_PANEL_ALT)])

    def _build_ui(self):
        container = ttk.Frame(self.root, style="App.TFrame", padding=(14, 14, 14, 14))
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=0)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(container, style="Sidebar.TFrame", padding=(14, 16, 14, 16), width=258)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(7, weight=1)

        ttk.Label(sidebar, text="Whisper\nTranscriber", style="SidebarTitle.TLabel", justify="left").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(sidebar, text="Local first. Fast on-device.\nOptional API diarization.", style="SidebarSubtitle.TLabel").grid(
            row=1, column=0, sticky="w", pady=(6, 14)
        )
        ttk.Separator(sidebar, orient="horizontal").grid(row=2, column=0, sticky="ew", pady=(0, 12))

        cards = ttk.Frame(sidebar, style="Sidebar.TFrame")
        cards.grid(row=3, column=0, sticky="ew")
        cards.columnconfigure(0, weight=1)

        def add_sidebar_card(row, label, value_var):
            card = ttk.Frame(cards, style="SidebarCard.TFrame", padding=(10, 8))
            card.grid(row=row, column=0, sticky="ew", pady=(0, 8))
            card.columnconfigure(0, weight=1)
            ttk.Label(card, text=label, style="SidebarMetaLabel.TLabel").grid(row=0, column=0, sticky="w")
            ttk.Label(card, textvariable=value_var, style="SidebarMetaValue.TLabel").grid(row=1, column=0, sticky="w", pady=(3, 0))

        add_sidebar_card(0, "Queue", self.sidebar_queue_var)
        add_sidebar_card(1, "Completed", self.sidebar_done_var)
        add_sidebar_card(2, "Backend", self.sidebar_backend_var)
        add_sidebar_card(3, "Timestamps", self.sidebar_timestamps_var)

        ttk.Label(
            sidebar,
            text="Tip: choose multilingual model\nfor Russian audio.",
            style="SidebarFoot.TLabel",
            justify="left",
        ).grid(row=8, column=0, sticky="sw", pady=(12, 0))

        main = ttk.Frame(container, style="App.TFrame", padding=(14, 0, 0, 0))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=2)
        main.rowconfigure(3, weight=1)

        header = ttk.Frame(main, style="TopCard.TFrame", padding=(16, 14, 16, 14))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(7, weight=1)

        ttk.Label(header, text="Transcription Workspace", style="TopTitle.TLabel").grid(row=0, column=0, columnspan=8, sticky="w")
        ttk.Label(
            header,
            text="Queue audio, choose backend, and export transcripts with optional timestamps.",
            style="TopSubtitle.TLabel",
        ).grid(row=1, column=0, columnspan=8, sticky="w", pady=(3, 12))

        ttk.Label(header, text="Source", style="TopMuted.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Radiobutton(
            header,
            text="Local (auto backend)",
            value="local",
            variable=self.source_var,
            command=self._set_model_options,
        ).grid(row=2, column=1, sticky="w", padx=(8, 16))
        ttk.Radiobutton(
            header,
            text="OpenAI API",
            value="openai",
            variable=self.source_var,
            command=self._set_model_options,
        ).grid(row=2, column=2, sticky="w", padx=(0, 16))

        ttk.Checkbutton(
            header,
            text="Include timestamps in output files",
            variable=self.timestamps_var,
            command=self._refresh_sidebar_stats,
        ).grid(row=2, column=3, columnspan=3, sticky="w")

        ttk.Label(header, text="Model", style="TopMuted.TLabel").grid(row=3, column=0, sticky="w", pady=(12, 0))
        self.model_combobox = ttk.Combobox(
            header,
            textvariable=self.model_display_var,
            state="readonly",
            width=74,
        )
        self.model_combobox.grid(row=3, column=1, columnspan=6, sticky="ew", padx=(8, 0), pady=(12, 0))
        self.model_combobox.bind("<<ComboboxSelected>>", lambda _event: None)

        controls = ttk.Frame(main, style="Toolbar.TFrame", padding=(0, 12, 0, 6))
        controls.grid(row=1, column=0, sticky="ew")
        controls.columnconfigure(10, weight=1)

        self.add_files_btn = ttk.Button(controls, text="Add Files", command=self.add_files, style="Secondary.TButton")
        self.add_files_btn.grid(row=0, column=0, padx=(0, 8))

        self.add_folder_btn = ttk.Button(controls, text="Add Folder", command=self.add_folder, style="Secondary.TButton")
        self.add_folder_btn.grid(row=0, column=1, padx=(0, 8))

        self.remove_btn = ttk.Button(controls, text="Remove Selected", command=self.remove_selected, style="Secondary.TButton")
        self.remove_btn.grid(row=0, column=2, padx=(0, 8))

        self.clear_btn = ttk.Button(controls, text="Clear", command=self.clear_queue, style="Secondary.TButton")
        self.clear_btn.grid(row=0, column=3, padx=(0, 16))

        self.start_btn = ttk.Button(controls, text="Start", command=self.start_processing, style="Primary.TButton")
        self.start_btn.grid(row=0, column=4, padx=(0, 8))

        self.stop_btn = ttk.Button(
            controls,
            text="Stop After Current",
            command=self.stop_processing,
            state="disabled",
            style="Danger.TButton",
        )
        self.stop_btn.grid(row=0, column=5, padx=(0, 8))

        self.open_folder_btn = ttk.Button(
            controls,
            text="Open Selected Folder",
            command=self.open_selected_folder,
            style="Secondary.TButton",
        )
        self.open_folder_btn.grid(row=0, column=6, padx=(0, 8))

        self.queue_frame = ttk.LabelFrame(main, text="Queue", style="Card.TLabelframe", padding=12)
        self.queue_frame.grid(row=2, column=0, pady=(4, 8), sticky="nsew")
        self.queue_frame.columnconfigure(0, weight=1)
        self.queue_frame.rowconfigure(5, weight=1)

        ttk.Label(self.queue_frame, textvariable=self.current_file_var, style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(self.queue_frame, textvariable=self.overall_var, style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 2))

        self.queue_progress = ttk.Progressbar(self.queue_frame, mode="determinate", maximum=100, value=0)
        self.queue_progress.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        self.current_progress = ttk.Progressbar(self.queue_frame, mode="indeterminate")
        self.current_progress.grid(row=3, column=0, sticky="ew", pady=(0, 10))

        self.drop_hint = ttk.Label(
            self.queue_frame,
            textvariable=self.drop_hint_var,
            style="DropHint.TLabel",
            anchor="center",
            justify="center",
        )
        self.drop_hint.grid(row=4, column=0, sticky="ew", pady=(0, 10))

        columns = ("name", "status")
        self.tree = ttk.Treeview(self.queue_frame, columns=columns, show="headings", selectmode="extended", height=14)
        self.tree.heading("name", text="File")
        self.tree.heading("status", text="Status")
        self.tree.column("name", width=760, anchor="w")
        self.tree.column("status", width=180, anchor="w")
        self.tree.grid(row=5, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(self.queue_frame, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=5, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)

        log_frame = ttk.LabelFrame(main, text="Logs", style="Card.TLabelframe", padding=12)
        log_frame.grid(row=3, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = ScrolledText(log_frame, height=12, wrap="word")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.configure(
            state="disabled",
            bg=COLOR_LOG_BG,
            fg=COLOR_LOG_TEXT,
            insertbackground=COLOR_LOG_TEXT,
            relief="flat",
            borderwidth=0,
            padx=8,
            pady=8,
            font=(FONT_MONO, 12),
        )

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _setup_drag_and_drop(self):
        if not self.drag_and_drop_enabled:
            self.drop_hint_var.set("Drag and drop unavailable. Use Add Files or Add Folder.")
            return

        for widget in (self.queue_frame, self.drop_hint, self.tree):
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<DropEnter>>", self._on_drop_enter)
            widget.dnd_bind("<<DropLeave>>", self._on_drop_leave)
            widget.dnd_bind("<<Drop>>", self._on_drop)

    def _on_drop_enter(self, _event):
        self.drop_hint.configure(style="DropHintActive.TLabel")
        self.drop_hint_var.set("Release to add files to the queue")
        return "copy"

    def _on_drop_leave(self, _event):
        self.drop_hint.configure(style="DropHint.TLabel")
        self.drop_hint_var.set("Drop audio files or folders here")
        return "copy"

    def _parse_drop_paths(self, raw_data):
        try:
            values = list(self.root.tk.splitlist(raw_data))
        except tk.TclError:
            values = [raw_data]

        paths = []
        for value in values:
            normalized = value.strip()
            if normalized.startswith("{") and normalized.endswith("}"):
                normalized = normalized[1:-1]
            normalized = os.path.expanduser(normalized)
            if normalized and normalized not in paths:
                paths.append(normalized)
        return paths

    def _add_dropped_paths(self, dropped_paths):
        added_files = 0
        added_folders = 0

        for dropped_path in dropped_paths:
            if os.path.isdir(dropped_path):
                added_folders += 1
                for file_name in sorted(os.listdir(dropped_path)):
                    file_path = os.path.join(dropped_path, file_name)
                    if os.path.isfile(file_path) and file_path.lower().endswith(core.AUDIO_EXTENSIONS):
                        before = len(self.file_paths)
                        self._add_file(file_path)
                        if len(self.file_paths) > before:
                            added_files += 1
            elif os.path.isfile(dropped_path) and dropped_path.lower().endswith(core.AUDIO_EXTENSIONS):
                before = len(self.file_paths)
                self._add_file(dropped_path)
                if len(self.file_paths) > before:
                    added_files += 1

        if added_files:
            self._update_overall_label(0, len(self.file_paths))
            folder_note = f" from {added_folders} folder(s)" if added_folders else ""
            self._log_line(f"Added {added_files} audio file(s){folder_note} via drag and drop.")
        else:
            self._log_line("Drag and drop ignored: no supported audio files found.")

    def _on_drop(self, event):
        self._on_drop_leave(event)
        dropped_paths = self._parse_drop_paths(event.data)
        self._add_dropped_paths(dropped_paths)
        return "copy"

    def _set_model_options(self):
        source = self.source_var.get()
        options = LOCAL_MODEL_OPTIONS if source == "local" else OPENAI_MODEL_OPTIONS
        self.model_display_to_value = {display: value for value, display in options}
        displays = [display for _value, display in options]
        self.model_combobox["values"] = displays
        if displays:
            if source == "local":
                preferred = next((display for value, display in options if value == "medium"), displays[0])
                self.model_display_var.set(preferred)
            else:
                self.model_display_var.set(displays[0])
        self._refresh_sidebar_stats()

    def _refresh_sidebar_stats(self):
        total = len(self.file_paths)
        done = 0
        for file_path, item in self.item_to_path.items():
            if not file_path:
                continue
            try:
                if self.tree.set(item, "status") == "Done":
                    done += 1
            except tk.TclError:
                continue

        backend = "Local backend" if self.source_var.get() == "local" else "OpenAI API"
        timestamps = "On" if self.timestamps_var.get() else "Off"
        self.sidebar_queue_var.set(f"{total} file(s) queued")
        self.sidebar_done_var.set(f"{done} completed")
        self.sidebar_backend_var.set(backend)
        self.sidebar_timestamps_var.set(f"{timestamps}")

    def _poll_log_queue(self):
        while True:
            try:
                chunk = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log(chunk)
        self.root.after(120, self._poll_log_queue)

    def _append_log(self, text):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _log_line(self, text):
        self.log_queue.put(text.rstrip() + "\n")

    def _set_controls_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        self.add_files_btn.configure(state=state)
        self.add_folder_btn.configure(state=state)
        self.remove_btn.configure(state=state)
        self.clear_btn.configure(state=state)
        self.start_btn.configure(state=state)
        self.model_combobox.configure(state="readonly" if enabled else "disabled")
        self.stop_btn.configure(state="disabled" if enabled else "normal")

    def _update_row_status(self, file_path, status):
        item = self.item_to_path.get(file_path)
        if item:
            self.tree.set(item, "status", status)
            self._refresh_sidebar_stats()

    def add_files(self):
        file_paths = filedialog.askopenfilenames(title="Select audio files", filetypes=AUDIO_FILETYPES)
        if not file_paths:
            return
        for file_path in file_paths:
            self._add_file(file_path)
        self._update_overall_label(0, len(self.file_paths))

    def add_folder(self):
        folder = filedialog.askdirectory(title="Select folder with audio files")
        if not folder:
            return
        added = 0
        for file_name in sorted(os.listdir(folder)):
            path = os.path.join(folder, file_name)
            if os.path.isfile(path) and path.lower().endswith(core.AUDIO_EXTENSIONS):
                self._add_file(path)
                added += 1
        self._log_line(f"Added {added} audio files from folder: {folder}")
        self._update_overall_label(0, len(self.file_paths))

    def _add_file(self, file_path):
        if file_path in self.file_paths:
            return
        self.file_paths.append(file_path)
        item = self.tree.insert("", "end", values=(os.path.basename(file_path), "Queued"))
        self.item_to_path[file_path] = item
        self._refresh_sidebar_stats()

    def remove_selected(self):
        selected = self.tree.selection()
        if not selected:
            return
        selected_paths = []
        for path, item_id in list(self.item_to_path.items()):
            if item_id in selected:
                selected_paths.append(path)

        for path in selected_paths:
            item_id = self.item_to_path.pop(path, None)
            if item_id:
                self.tree.delete(item_id)
            if path in self.file_paths:
                self.file_paths.remove(path)
        self._update_overall_label(0, len(self.file_paths))
        self._refresh_sidebar_stats()

    def clear_queue(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Processing", "Stop current run before clearing queue.")
            return
        self.file_paths.clear()
        self.item_to_path.clear()
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.current_file_var.set("Current file: -")
        self._update_overall_label(0, 0)
        self.queue_progress.configure(maximum=100, value=0)
        self._refresh_sidebar_stats()

    def open_selected_folder(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo("Open Folder", "Select a file in the queue first.")
            return
        selected_item = selected[0]
        selected_path = None
        for path, item_id in self.item_to_path.items():
            if item_id == selected_item:
                selected_path = path
                break
        if not selected_path:
            return
        folder = os.path.dirname(selected_path)
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", folder], check=False)
            elif os.name == "nt":
                os.startfile(folder)  # type: ignore[attr-defined]
            else:
                subprocess.run(["xdg-open", folder], check=False)
        except Exception as exc:
            messagebox.showerror("Open Folder", f"Failed to open folder:\n{exc}")

    def stop_processing(self):
        if not self.worker_thread or not self.worker_thread.is_alive():
            return
        self.stop_event.set()
        self._log_line("Stop requested: processing will stop after current file.")

    def _update_overall_label(self, done, total):
        self.overall_var.set(f"Queue progress: {done}/{total}")
        self.queue_progress.configure(maximum=max(total, 1), value=done)
        self._refresh_sidebar_stats()

    def start_processing(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Processing", "Transcription is already running.")
            return
        if not self.file_paths:
            messagebox.showwarning("Queue is empty", "Add at least one audio file.")
            return

        model_display = self.model_display_var.get().strip()
        if not model_display:
            messagebox.showwarning("Model", "Choose a model first.")
            return
        model_value = self.model_display_to_value.get(model_display)
        if not model_value:
            messagebox.showwarning("Model", "Model mapping is invalid. Re-select model.")
            return

        for path in self.file_paths:
            self._update_row_status(path, "Queued")

        auto_install_mlx = None
        if self.source_var.get() == "local" and sys.platform == "darwin" and not core.has_module("mlx_whisper"):
            auto_install_mlx = messagebox.askyesno(
                "Install mlx-whisper",
                "mlx-whisper is not installed.\n\nInstall now for faster local transcription on Mac?",
            )

        include_timestamps = bool(self.timestamps_var.get())
        source = self.source_var.get()
        files_snapshot = list(self.file_paths)
        self.stop_event.clear()

        self.current_file_var.set("Current file: preparing backend...")
        self._update_overall_label(0, len(files_snapshot))
        self.current_progress.start(10)
        self._set_controls_enabled(False)

        self.worker_thread = threading.Thread(
            target=self._worker_run,
            args=(source, model_value, include_timestamps, files_snapshot, auto_install_mlx),
            daemon=True,
        )
        self.worker_thread.start()

    def _worker_run(self, source, model_value, include_timestamps, files, auto_install_mlx):
        log_writer = QueueLogWriter(self.log_queue)
        done = 0
        total = len(files)

        try:
            with contextlib.redirect_stdout(log_writer), contextlib.redirect_stderr(log_writer):
                if source == "local":
                    transcribe_function = core.create_local_transcribe_function(
                        model_value,
                        include_timestamps,
                        interactive_prompt=False,
                        auto_install_mlx=auto_install_mlx,
                    )
                else:
                    transcribe_function = lambda file_path: core.transcribe_audio_openai(
                        file_path,
                        model_value,
                        include_timestamps,
                    )

            for file_path in files:
                if self.stop_event.is_set():
                    self.root.after(0, lambda p=file_path: self._update_row_status(p, "Skipped"))
                    continue

                self.root.after(
                    0,
                    lambda p=file_path: self.current_file_var.set(f"Current file: {os.path.basename(p)}"),
                )
                self.root.after(0, lambda p=file_path: self._update_row_status(p, "Processing"))

                try:
                    with contextlib.redirect_stdout(log_writer), contextlib.redirect_stderr(log_writer):
                        transcribe_function(file_path)
                    done += 1
                    self.root.after(0, lambda p=file_path: self._update_row_status(p, "Done"))
                    self.root.after(0, lambda d=done, t=total: self._update_overall_label(d, t))
                except Exception as exc:
                    done += 1
                    error_message = f"Error: {exc}"
                    self.root.after(0, lambda p=file_path, e=error_message: self._update_row_status(p, e))
                    self.root.after(0, lambda d=done, t=total: self._update_overall_label(d, t))
                    self._log_line(f"Error for {file_path}: {exc}")
                    self._log_line(traceback.format_exc())

            if self.stop_event.is_set():
                self._log_line("Run finished with stop request.")
            else:
                self._log_line("Run finished successfully.")
        except Exception as exc:
            self._log_line(f"Failed to initialize transcription backend: {exc}")
            self._log_line(traceback.format_exc())
        finally:
            self.root.after(0, self._finish_run_ui)

    def _finish_run_ui(self):
        self.current_progress.stop()
        self._set_controls_enabled(True)
        self.current_file_var.set("Current file: -")
        self._refresh_sidebar_stats()

    def on_close(self):
        if self.worker_thread and self.worker_thread.is_alive():
            if not messagebox.askyesno("Exit", "Transcription is running. Stop after current file and close?"):
                return
            self.stop_event.set()
        self.root.destroy()


def main():
    root = TkinterDnD.Tk() if TkinterDnD else tk.Tk()
    app = TranscriptionApp(root)
    app._log_line("UI started.")
    root.mainloop()


if __name__ == "__main__":
    main()
