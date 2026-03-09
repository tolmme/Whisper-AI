import contextlib
import os
import queue
import subprocess
import sys
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk

import main as core

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    DND_FILES = None
    TkinterDnD = None


LOCAL_MODEL_OPTIONS = [
    ("small.en", "ENG Small (small.en, ~2 GB RAM)"),
    ("medium.en", "ENG Medium (medium.en, ~5 GB RAM)"),
    ("small", "Multilingual Small (small, ~2 GB RAM)"),
    ("medium", "Multilingual Medium (medium, ~5 GB RAM)"),
    ("large-v3", "Multilingual Large-v3 (large-v3, ~9-10 GB RAM)"),
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

# ── Codex-inspired palette ──────────────────────────────────────
_MAC = sys.platform == "darwin"

WIN_BG = "#EDEDF0"           # Window / sidebar (light cool gray)
CONTENT_BG = "#FFFFFF"        # White content panel
SURFACE = "#F5F5F7"           # Elevated surface inside content
RADIUS = 12                   # Content panel corner radius

TEXT_1 = "#1D1D1F"
TEXT_2 = "#86868B"
TEXT_3 = "#AEAEB2"

ACCENT = "#007AFF"
ACCENT_HOVER = "#0055D4"
DANGER = "#FF3B30"
SUCCESS = "#34C759"

BORDER = "#E5E5EA"
SEP = "#D8D8DC"

LOG_BG = "#1C1C1E"
LOG_FG = "#E5E5EA"

FONT = "SF Pro" if _MAC else "Segoe UI"
FONT_MONO = "SF Mono" if _MAC else "Consolas"


class QueueLogWriter:
    def __init__(self, output_queue):
        self.output_queue = output_queue

    def write(self, text):
        if text:
            self.output_queue.put(text)

    def flush(self):
        return None


def _create_root():
    """Create root window with optional drag-and-drop support."""
    if TkinterDnD:
        try:
            class _DnDCTk(ctk.CTk, TkinterDnD.DnDWrapper):
                def __init__(self):
                    super().__init__()
                    self.TkdndVersion = TkinterDnD._require(self)
            return _DnDCTk(), True
        except Exception:
            pass
    return ctk.CTk(), False


class TranscriptionApp:
    def __init__(self, root, dnd_ok=False):
        self.root = root
        self.dnd_ok = dnd_ok
        root.title("Whisper Transcriber")
        root.geometry("1200x780")
        root.minsize(1060, 700)

        ctk.set_appearance_mode("light")

        self.file_paths: list[str] = []
        self.item_to_path: dict[str, str] = {}
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.log_queue: queue.Queue[str] = queue.Queue()

        self.source_var = tk.StringVar(value="local")
        self.timestamps_var = tk.BooleanVar(value=True)
        self.current_file_var = tk.StringVar(value="")
        self.overall_var = tk.StringVar(value="0 / 0")
        self.model_display_var = tk.StringVar()
        self.sidebar_queue_var = tk.StringVar(value="0")
        self.sidebar_done_var = tk.StringVar(value="0")
        self.sidebar_backend_var = tk.StringVar(value="Local")
        self.sidebar_timestamps_var = tk.StringVar(value="On")
        self.drop_hint_var = tk.StringVar(value="Drop audio files or folders here")

        self.model_display_to_value: dict[str, str] = {}

        self._build_ui()
        self._style_treeview()
        self._set_model_options()
        self._setup_drag_and_drop()
        self._poll_log_queue()

    # ── fonts (cached) ─────────────────────────────────────────

    def _f(self, size=13, weight="normal"):
        return ctk.CTkFont(family=FONT, size=size, weight=weight)

    def _fm(self, size=12):
        return ctk.CTkFont(family=FONT_MONO, size=size)

    # ── UI ─────────────────────────────────────────────────────

    def _build_ui(self):
        root = self.root
        root.configure(fg_color=WIN_BG)
        root.grid_columnconfigure(0, weight=0)
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)

        # ── Sidebar (transparent → shows WIN_BG) ──────────────
        sb = ctk.CTkFrame(root, fg_color="transparent", width=220)
        sb.grid(row=0, column=0, sticky="ns", padx=(18, 0), pady=(14, 18))
        sb.grid_propagate(False)
        self._build_sidebar(sb)

        # ── Content (white rounded panel) ──────────────────────
        ct = ctk.CTkFrame(root, fg_color=CONTENT_BG, corner_radius=RADIUS)
        ct.grid(row=0, column=1, sticky="nsew", padx=(10, 12), pady=(8, 12))
        self._build_content(ct)

        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_sidebar(self, sb):
        sb.grid_columnconfigure(0, weight=1)
        sb.grid_columnconfigure(1, weight=0)
        sb.grid_rowconfigure(10, weight=1)

        ctk.CTkLabel(
            sb, text="Whisper\nTranscriber", font=self._f(20, "bold"),
            text_color=TEXT_1, justify="left", anchor="w",
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        ctk.CTkLabel(
            sb, text="Audio transcription tool", font=self._f(11),
            text_color=TEXT_2, anchor="w",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 20))

        ctk.CTkFrame(sb, fg_color=SEP, height=1).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(0, 16))

        ctk.CTkLabel(
            sb, text="STATUS", font=self._f(10, "bold"),
            text_color=TEXT_3, anchor="w",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 12))

        def stat(r, label, var):
            ctk.CTkLabel(sb, text=label, font=self._f(12),
                         text_color=TEXT_2, anchor="w").grid(
                row=r, column=0, sticky="w", pady=(0, 10))
            ctk.CTkLabel(sb, textvariable=var, font=self._f(13),
                         text_color=TEXT_1, anchor="e").grid(
                row=r, column=1, sticky="e", pady=(0, 10))

        stat(4, "Queue", self.sidebar_queue_var)
        stat(5, "Completed", self.sidebar_done_var)
        stat(6, "Backend", self.sidebar_backend_var)
        stat(7, "Timestamps", self.sidebar_timestamps_var)

        ctk.CTkLabel(
            sb, text="Tip: use multilingual model\nfor Russian audio.",
            font=self._f(11), text_color=TEXT_3,
            justify="left", anchor="sw",
        ).grid(row=11, column=0, columnspan=2, sticky="sw")

    def _build_content(self, ct):
        ct.grid_columnconfigure(0, weight=1)
        ct.grid_rowconfigure(4, weight=3)
        ct.grid_rowconfigure(6, weight=1)
        px = 22  # horizontal padding inside content

        # Row 0 — Source + timestamps ───────────────────────────
        r0 = ctk.CTkFrame(ct, fg_color="transparent")
        r0.grid(row=0, column=0, sticky="ew", padx=px, pady=(18, 0))
        r0.grid_columnconfigure(5, weight=1)

        ctk.CTkLabel(r0, text="Source", font=self._f(12, "bold"),
                     text_color=TEXT_2).grid(row=0, column=0, sticky="w")

        ctk.CTkRadioButton(
            r0, text="Local (auto)", variable=self.source_var,
            value="local", command=self._set_model_options,
            fg_color=ACCENT, border_color=BORDER,
            font=self._f(12),
        ).grid(row=0, column=1, sticky="w", padx=(12, 14))

        ctk.CTkRadioButton(
            r0, text="OpenAI API", variable=self.source_var,
            value="openai", command=self._set_model_options,
            fg_color=ACCENT, border_color=BORDER,
            font=self._f(12),
        ).grid(row=0, column=2, sticky="w", padx=(0, 24))

        ctk.CTkCheckBox(
            r0, text="Include timestamps",
            variable=self.timestamps_var,
            command=self._refresh_sidebar_stats,
            fg_color=ACCENT, border_color=BORDER,
            font=self._f(12),
        ).grid(row=0, column=3, sticky="w")

        # Row 1 — Model ────────────────────────────────────────
        r1 = ctk.CTkFrame(ct, fg_color="transparent")
        r1.grid(row=1, column=0, sticky="ew", padx=px, pady=(10, 0))
        r1.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(r1, text="Model", font=self._f(12, "bold"),
                     text_color=TEXT_2).grid(row=0, column=0, sticky="w")

        self.model_combobox = ctk.CTkComboBox(
            r1, variable=self.model_display_var, state="readonly",
            fg_color=CONTENT_BG, border_color=BORDER,
            button_color="#E5E5EA", button_hover_color="#D1D1D6",
            dropdown_fg_color=CONTENT_BG, dropdown_hover_color=SURFACE,
            corner_radius=8, font=self._f(13),
        )
        self.model_combobox.grid(row=0, column=1, sticky="ew", padx=(12, 0))

        # Row 2 — Separator ────────────────────────────────────
        ctk.CTkFrame(ct, fg_color=BORDER, height=1).grid(
            row=2, column=0, sticky="ew", padx=px, pady=(16, 10))

        # Row 3 — Toolbar ──────────────────────────────────────
        tb = ctk.CTkFrame(ct, fg_color="transparent")
        tb.grid(row=3, column=0, sticky="ew", padx=px)
        tb.grid_columnconfigure(7, weight=1)

        bf = self._f(12)
        bfb = self._f(12, "bold")

        def sec(parent, text, cmd):
            return ctk.CTkButton(
                parent, text=text, command=cmd,
                fg_color=SURFACE, text_color=TEXT_1,
                hover_color="#ECECEC", border_width=1, border_color=BORDER,
                corner_radius=8, height=32, font=bf,
            )

        self.add_files_btn = sec(tb, "Add Files", self.add_files)
        self.add_files_btn.grid(row=0, column=0, padx=(0, 5))

        self.add_folder_btn = sec(tb, "Add Folder", self.add_folder)
        self.add_folder_btn.grid(row=0, column=1, padx=(0, 5))

        self.remove_btn = sec(tb, "Remove", self.remove_selected)
        self.remove_btn.grid(row=0, column=2, padx=(0, 5))

        self.clear_btn = sec(tb, "Clear", self.clear_queue)
        self.clear_btn.grid(row=0, column=3, padx=(0, 18))

        self.start_btn = ctk.CTkButton(
            tb, text="Start", command=self.start_processing,
            fg_color=ACCENT, text_color="#FFF", hover_color=ACCENT_HOVER,
            corner_radius=8, height=32, font=bfb,
        )
        self.start_btn.grid(row=0, column=4, padx=(0, 5))

        self.stop_btn = ctk.CTkButton(
            tb, text="Stop After Current", command=self.stop_processing,
            fg_color="#FFF0EF", text_color=DANGER, hover_color="#FFE0DD",
            border_width=1, border_color="#FFCCC9",
            corner_radius=8, height=32, font=bf, state="disabled",
        )
        self.stop_btn.grid(row=0, column=5, padx=(0, 5))

        self.open_folder_btn = sec(tb, "Open Folder", self.open_selected_folder)
        self.open_folder_btn.grid(row=0, column=6)

        # Row 4 — Queue area ───────────────────────────────────
        qa = ctk.CTkFrame(ct, fg_color="transparent")
        qa.grid(row=4, column=0, sticky="nsew", padx=px, pady=(10, 0))
        qa.grid_columnconfigure(0, weight=1)
        qa.grid_rowconfigure(4, weight=1)

        # info row
        info = ctk.CTkFrame(qa, fg_color="transparent")
        info.grid(row=0, column=0, sticky="ew")
        info.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(info, textvariable=self.current_file_var,
                     font=self._f(12), text_color=TEXT_2,
                     anchor="w").grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(info, textvariable=self.overall_var,
                     font=self._f(12), text_color=TEXT_2,
                     anchor="e").grid(row=0, column=1, sticky="e")

        # progress bars
        self.queue_progress = ctk.CTkProgressBar(
            qa, fg_color="#E5E5EA", progress_color=ACCENT,
            height=4, corner_radius=2)
        self.queue_progress.grid(row=1, column=0, sticky="ew", pady=(6, 2))
        self.queue_progress.set(0)

        self.current_progress = ctk.CTkProgressBar(
            qa, fg_color="#E5E5EA", progress_color=ACCENT,
            height=4, corner_radius=2, mode="indeterminate",
            indeterminate_speed=0.8)
        self.current_progress.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        # drop zone (rounded)
        self.drop_frame = ctk.CTkFrame(
            qa, fg_color=SURFACE, corner_radius=8,
            border_width=1, border_color=BORDER, height=36)
        self.drop_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        self.drop_frame.grid_propagate(False)
        self.drop_frame.grid_columnconfigure(0, weight=1)
        self.drop_frame.grid_rowconfigure(0, weight=1)

        self.drop_label = ctk.CTkLabel(
            self.drop_frame, textvariable=self.drop_hint_var,
            font=self._f(12), text_color=TEXT_3)
        self.drop_label.grid(row=0, column=0)

        # treeview
        tree_wrap = ctk.CTkFrame(qa, fg_color=CONTENT_BG, corner_radius=0)
        tree_wrap.grid(row=4, column=0, sticky="nsew")
        tree_wrap.grid_columnconfigure(0, weight=1)
        tree_wrap.grid_rowconfigure(0, weight=1)

        cols = ("name", "status")
        self.tree = ttk.Treeview(
            tree_wrap, columns=cols, show="headings",
            selectmode="extended", height=10)
        self.tree.heading("name", text="File")
        self.tree.heading("status", text="Status")
        self.tree.column("name", width=660, anchor="w")
        self.tree.column("status", width=120, anchor="w")
        self.tree.grid(row=0, column=0, sticky="nsew")

        vsb = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.tree.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=vsb.set)

        # Row 5 — Separator ────────────────────────────────────
        ctk.CTkFrame(ct, fg_color=BORDER, height=1).grid(
            row=5, column=0, sticky="ew", padx=px, pady=(10, 6))

        # Row 6 — Log area ─────────────────────────────────────
        la = ctk.CTkFrame(ct, fg_color="transparent")
        la.grid(row=6, column=0, sticky="nsew", padx=px, pady=(0, 16))
        la.grid_columnconfigure(0, weight=1)
        la.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(la, text="OUTPUT", font=self._f(11, "bold"),
                     text_color=TEXT_2, anchor="w").grid(
            row=0, column=0, sticky="w", pady=(0, 4))

        self.log_text = ctk.CTkTextbox(
            la, fg_color=LOG_BG, text_color=LOG_FG,
            font=self._fm(12), corner_radius=8,
            border_width=0, state="disabled")
        self.log_text.grid(row=1, column=0, sticky="nsew")

    # ── Treeview styling (ttk) ─────────────────────────────────

    def _style_treeview(self):
        s = ttk.Style()
        if "clam" in s.theme_names():
            s.theme_use("clam")
        s.configure(
            "Treeview", background=CONTENT_BG, fieldbackground=CONTENT_BG,
            foreground=TEXT_1, rowheight=30, borderwidth=0,
            font=(FONT, 12),
        )
        s.map("Treeview",
              background=[("selected", "#DCEEFF")],
              foreground=[("selected", TEXT_1)])
        s.configure(
            "Treeview.Heading", background=SURFACE, foreground=TEXT_2,
            bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
            relief="flat", font=(FONT, 11, "bold"),
        )
        s.map("Treeview.Heading", background=[("active", SURFACE)])

    # ── Drag-and-drop ──────────────────────────────────────────

    def _setup_drag_and_drop(self):
        if not self.dnd_ok:
            self.drop_hint_var.set(
                "Drag and drop unavailable. Use Add Files or Add Folder.")
            return
        # Treeview is a standard tk widget — DnD always works
        for w in (self.tree,):
            w.drop_target_register(DND_FILES)
            w.dnd_bind("<<DropEnter>>", self._on_drop_enter)
            w.dnd_bind("<<DropLeave>>", self._on_drop_leave)
            w.dnd_bind("<<Drop>>", self._on_drop)
        # Try CTk widgets (may or may not work)
        for w in (self.drop_frame, self.drop_label):
            try:
                w.drop_target_register(DND_FILES)
                w.dnd_bind("<<DropEnter>>", self._on_drop_enter)
                w.dnd_bind("<<DropLeave>>", self._on_drop_leave)
                w.dnd_bind("<<Drop>>", self._on_drop)
            except Exception:
                pass

    def _on_drop_enter(self, _event):
        self.drop_frame.configure(fg_color="#E3EFFD", border_color=ACCENT)
        self.drop_hint_var.set("Release to add files to the queue")
        return "copy"

    def _on_drop_leave(self, _event):
        self.drop_frame.configure(fg_color=SURFACE, border_color=BORDER)
        self.drop_hint_var.set("Drop audio files or folders here")
        return "copy"

    def _parse_drop_paths(self, raw_data):
        try:
            values = list(self.root.tk.splitlist(raw_data))
        except tk.TclError:
            values = [raw_data]
        paths: list[str] = []
        for v in values:
            n = v.strip()
            if n.startswith("{") and n.endswith("}"):
                n = n[1:-1]
            n = os.path.expanduser(n)
            if n and n not in paths:
                paths.append(n)
        return paths

    def _add_dropped_paths(self, dropped):
        added_files = 0
        added_folders = 0
        for p in dropped:
            if os.path.isdir(p):
                added_folders += 1
                for fn in sorted(os.listdir(p)):
                    fp = os.path.join(p, fn)
                    if os.path.isfile(fp) and fp.lower().endswith(
                            core.AUDIO_EXTENSIONS):
                        before = len(self.file_paths)
                        self._add_file(fp)
                        if len(self.file_paths) > before:
                            added_files += 1
            elif os.path.isfile(p) and p.lower().endswith(
                    core.AUDIO_EXTENSIONS):
                before = len(self.file_paths)
                self._add_file(p)
                if len(self.file_paths) > before:
                    added_files += 1
        if added_files:
            self._update_overall_label(0, len(self.file_paths))
            note = f" from {added_folders} folder(s)" if added_folders else ""
            self._log_line(
                f"Added {added_files} audio file(s){note} via drag and drop.")
        else:
            self._log_line(
                "Drag and drop ignored: no supported audio files found.")

    def _on_drop(self, event):
        self._on_drop_leave(event)
        self._add_dropped_paths(self._parse_drop_paths(event.data))
        return "copy"

    # ── Model / sidebar helpers ────────────────────────────────

    def _set_model_options(self):
        source = self.source_var.get()
        options = (LOCAL_MODEL_OPTIONS if source == "local"
                   else OPENAI_MODEL_OPTIONS)
        self.model_display_to_value = {d: v for v, d in options}
        displays = [d for _, d in options]
        self.model_combobox.configure(values=displays)
        if displays:
            if source == "local":
                preferred = next(
                    (d for v, d in options if v == "medium"), displays[0])
                self.model_combobox.set(preferred)
            else:
                self.model_combobox.set(displays[0])
        self._refresh_sidebar_stats()

    def _refresh_sidebar_stats(self):
        total = len(self.file_paths)
        done = 0
        for fp, item in self.item_to_path.items():
            if not fp:
                continue
            try:
                if self.tree.set(item, "status") == "Done":
                    done += 1
            except tk.TclError:
                continue
        self.sidebar_queue_var.set(str(total))
        self.sidebar_done_var.set(str(done))
        self.sidebar_backend_var.set(
            "Local" if self.source_var.get() == "local" else "OpenAI API")
        self.sidebar_timestamps_var.set(
            "On" if self.timestamps_var.get() else "Off")

    # ── Log polling ────────────────────────────────────────────

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
        self.log_text._textbox.see("end")
        self.log_text.configure(state="disabled")

    def _log_line(self, text):
        self.log_queue.put(text.rstrip() + "\n")

    # ── Controls state ─────────────────────────────────────────

    def _set_controls_enabled(self, enabled):
        st = "normal" if enabled else "disabled"
        self.add_files_btn.configure(state=st)
        self.add_folder_btn.configure(state=st)
        self.remove_btn.configure(state=st)
        self.clear_btn.configure(state=st)
        self.start_btn.configure(state=st)
        self.model_combobox.configure(state="readonly" if enabled else "disabled")
        self.stop_btn.configure(state="disabled" if enabled else "normal")

    def _update_row_status(self, file_path, status):
        item = self.item_to_path.get(file_path)
        if item:
            self.tree.set(item, "status", status)
            self._refresh_sidebar_stats()

    # ── Queue management ───────────────────────────────────────

    def add_files(self):
        fps = filedialog.askopenfilenames(
            title="Select audio files", filetypes=AUDIO_FILETYPES)
        if not fps:
            return
        for fp in fps:
            self._add_file(fp)
        self._update_overall_label(0, len(self.file_paths))

    def add_folder(self):
        folder = filedialog.askdirectory(
            title="Select folder with audio files")
        if not folder:
            return
        added = 0
        for fn in sorted(os.listdir(folder)):
            p = os.path.join(folder, fn)
            if os.path.isfile(p) and p.lower().endswith(core.AUDIO_EXTENSIONS):
                self._add_file(p)
                added += 1
        self._log_line(f"Added {added} audio files from folder: {folder}")
        self._update_overall_label(0, len(self.file_paths))

    def _add_file(self, file_path):
        if file_path in self.file_paths:
            return
        self.file_paths.append(file_path)
        item = self.tree.insert(
            "", "end", values=(os.path.basename(file_path), "Queued"))
        self.item_to_path[file_path] = item
        self._refresh_sidebar_stats()

    def remove_selected(self):
        selected = self.tree.selection()
        if not selected:
            return
        to_remove = [
            p for p, iid in list(self.item_to_path.items())
            if iid in selected
        ]
        for p in to_remove:
            iid = self.item_to_path.pop(p, None)
            if iid:
                self.tree.delete(iid)
            if p in self.file_paths:
                self.file_paths.remove(p)
        self._update_overall_label(0, len(self.file_paths))
        self._refresh_sidebar_stats()

    def clear_queue(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo(
                "Processing", "Stop current run before clearing queue.")
            return
        self.file_paths.clear()
        self.item_to_path.clear()
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.current_file_var.set("")
        self._update_overall_label(0, 0)
        self.queue_progress.set(0)
        self._refresh_sidebar_stats()

    def open_selected_folder(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo(
                "Open Folder", "Select a file in the queue first.")
            return
        sel_path = None
        for p, iid in self.item_to_path.items():
            if iid == selected[0]:
                sel_path = p
                break
        if not sel_path:
            return
        folder = os.path.dirname(sel_path)
        try:
            if _MAC:
                subprocess.run(["open", folder], check=False)
            elif os.name == "nt":
                os.startfile(folder)  # type: ignore[attr-defined]
            else:
                subprocess.run(["xdg-open", folder], check=False)
        except Exception as exc:
            messagebox.showerror(
                "Open Folder", f"Failed to open folder:\n{exc}")

    # ── Processing ─────────────────────────────────────────────

    def stop_processing(self):
        if not self.worker_thread or not self.worker_thread.is_alive():
            return
        self.stop_event.set()
        self._log_line("Stop requested: will stop after current file.")

    def _update_overall_label(self, done, total):
        self.overall_var.set(f"{done} / {total}")
        self.queue_progress.set(done / max(total, 1))
        self._refresh_sidebar_stats()

    def start_processing(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo(
                "Processing", "Transcription is already running.")
            return
        if not self.file_paths:
            messagebox.showwarning(
                "Queue is empty", "Add at least one audio file.")
            return

        model_display = self.model_display_var.get().strip()
        if not model_display:
            messagebox.showwarning("Model", "Choose a model first.")
            return
        model_value = self.model_display_to_value.get(model_display)
        if not model_value:
            messagebox.showwarning(
                "Model", "Model mapping is invalid. Re-select model.")
            return

        for p in self.file_paths:
            self._update_row_status(p, "Queued")

        auto_install_mlx = None
        if (self.source_var.get() == "local" and _MAC
                and not core.has_module("mlx_whisper")):
            auto_install_mlx = messagebox.askyesno(
                "Install mlx-whisper",
                "mlx-whisper is not installed.\n\n"
                "Install now for faster local transcription on Mac?",
            )

        include_timestamps = bool(self.timestamps_var.get())
        source = self.source_var.get()
        files_snapshot = list(self.file_paths)
        self.stop_event.clear()

        self.current_file_var.set("Preparing backend\u2026")
        self._update_overall_label(0, len(files_snapshot))
        self.current_progress.start()
        self._set_controls_enabled(False)

        self.worker_thread = threading.Thread(
            target=self._worker_run,
            args=(source, model_value, include_timestamps,
                  files_snapshot, auto_install_mlx),
            daemon=True,
        )
        self.worker_thread.start()

    def _worker_run(self, source, model_value, include_timestamps,
                    files, auto_install_mlx):
        log_writer = QueueLogWriter(self.log_queue)
        done = 0
        total = len(files)

        try:
            with (contextlib.redirect_stdout(log_writer),
                  contextlib.redirect_stderr(log_writer)):
                if source == "local":
                    transcribe_function = \
                        core.create_local_transcribe_function(
                            model_value, include_timestamps,
                            interactive_prompt=False,
                            auto_install_mlx=auto_install_mlx)
                else:
                    transcribe_function = (
                        lambda fp: core.transcribe_audio_openai(
                            fp, model_value, include_timestamps))

            for fp in files:
                if self.stop_event.is_set():
                    self.root.after(
                        0, lambda p=fp:
                        self._update_row_status(p, "Skipped"))
                    continue

                self.root.after(
                    0, lambda p=fp:
                    self.current_file_var.set(os.path.basename(p)))
                self.root.after(
                    0, lambda p=fp:
                    self._update_row_status(p, "Processing"))

                try:
                    with (contextlib.redirect_stdout(log_writer),
                          contextlib.redirect_stderr(log_writer)):
                        transcribe_function(fp)
                    done += 1
                    self.root.after(
                        0, lambda p=fp:
                        self._update_row_status(p, "Done"))
                    self.root.after(
                        0, lambda d=done, t=total:
                        self._update_overall_label(d, t))
                except Exception as exc:
                    done += 1
                    err = f"Error: {exc}"
                    self.root.after(
                        0, lambda p=fp, e=err:
                        self._update_row_status(p, e))
                    self.root.after(
                        0, lambda d=done, t=total:
                        self._update_overall_label(d, t))
                    self._log_line(f"Error for {fp}: {exc}")
                    self._log_line(traceback.format_exc())

            if self.stop_event.is_set():
                self._log_line("Run finished with stop request.")
            else:
                self._log_line("Run finished successfully.")
        except Exception as exc:
            self._log_line(
                f"Failed to initialize transcription backend: {exc}")
            self._log_line(traceback.format_exc())
        finally:
            self.root.after(0, self._finish_run_ui)

    def _finish_run_ui(self):
        self.current_progress.stop()
        self._set_controls_enabled(True)
        self.current_file_var.set("")
        self._refresh_sidebar_stats()

    def on_close(self):
        if self.worker_thread and self.worker_thread.is_alive():
            if not messagebox.askyesno(
                    "Exit",
                    "Transcription is running. "
                    "Stop after current file and close?"):
                return
            self.stop_event.set()
        self.root.destroy()


def main():
    root, dnd_ok = _create_root()
    app = TranscriptionApp(root, dnd_ok)
    app._log_line("UI started.")
    root.mainloop()


if __name__ == "__main__":
    main()
