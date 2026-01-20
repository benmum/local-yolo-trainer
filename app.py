import os
import re
import json
import time
import queue
import shutil
import threading
import subprocess
from dataclasses import dataclass, field

from flask import Flask, render_template, request, jsonify, Response, send_from_directory

import tkinter as tk
from tkinter import filedialog

app = Flask(__name__)


# Simple in-memory state

@dataclass
class TrainState:
    dataset_path: str | None = None
    output_path: str | None = None

    status: str = "idle"  
    progress_pct: int = 0
    current_epoch: int = 0
    total_epochs: int = 0

    run_name: str | None = None

    
    best_pt_path: str | None = None

    
    exported_model_path: str | None = None

    last_error: str | None = None
    logs: queue.Queue = field(default_factory=queue.Queue)  


STATE = TrainState()
LOCK = threading.Lock()


EPOCH_RE = re.compile(r"(?:^|\s)Epoch\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
EPOCH_LINESTART_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)(?:\s|$)")


# Helpers



def push_log(line: str):
    
    try:
        STATE.logs.put(line)
    except Exception:
        pass

def _runs_base_dir() -> str:
    
    with LOCK:
        project_dir = STATE.output_path
    return project_dir if project_dir else os.path.join(os.getcwd(), "runs", "detect")

def pick_directory_dialog(title: str) -> str | None:
    
    root = tk.Tk()
    root.withdraw()
    root.update()
    try:
        path = filedialog.askdirectory(title=title, mustexist=True)
    finally:
        root.destroy()
    return path if path else None


def pick_save_file_dialog(title: str, default_name: str) -> str | None:
    
    root = tk.Tk()
    root.withdraw()
    root.update()
    try:
        path = filedialog.asksaveasfilename(
            title=title,
            initialfile=default_name,
            defaultextension=".pt",
            filetypes=[("PyTorch Weights", "*.pt"), ("All files", "*.*")]
        )
    finally:
        root.destroy()
    return path if path else None


def validate_yolo_dataset(dataset_root: str) -> tuple[bool, str]:
    
    data_yaml = os.path.join(dataset_root, "data.yaml")
    if os.path.isfile(data_yaml):
        return True, "Found data.yaml"

    images_train = os.path.join(dataset_root, "images", "train")
    labels_train = os.path.join(dataset_root, "labels", "train")
    if os.path.isdir(images_train) and os.path.isdir(labels_train):
        return True, "Found images/train and labels/train"

    return False, (
        "Folder doesn't look like a YOLO dataset.\n"
        "Pick the dataset ROOT folder that contains either:\n"
        " - data.yaml\n"
        "OR\n"
        " - images/train and labels/train\n"
        f"\nSelected: {dataset_root}\n"
        f"Expected: {images_train} and {labels_train}"
    )


def _safe_run_name(name: str) -> str:
    name = (name or "").strip() or f"train_{int(time.time())}"
    name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
    return name[:80]

def _run_dir_for(name: str) -> str:
    
    with LOCK:
        project_dir = STATE.output_path
    base = project_dir if project_dir else os.path.join(os.getcwd(), "runs", "detect")
    return os.path.join(base, name)

def _safe_join(base: str, *paths: str) -> str:
    
    base = os.path.abspath(base)
    candidate = os.path.abspath(os.path.join(base, *paths))
    if not candidate.startswith(base + os.sep) and candidate != base:
        raise ValueError("Invalid path")
    return candidate

def _read_last_metrics_csv(csv_path: str) -> dict:
    
    if not os.path.isfile(csv_path):
        return {}

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if len(lines) < 2:
        return {}

    header = [h.strip() for h in lines[0].split(",")]
    last = [v.strip() for v in lines[-1].split(",")]

    
    if len(last) < len(header):
        last += [""] * (len(header) - len(last))
    last = last[:len(header)]

    out = {}
    for k, v in zip(header, last):
        
        try:
            out[k] = float(v)
        except Exception:
            out[k] = v
    return out

# Training worker
def run_training_job(
    data_arg: str,
    epochs: int,
    imgsz: int,
    model: str,
    project_dir: str | None,
    run_name: str,
):
    
    with LOCK:
        STATE.status = "running"
        STATE.progress_pct = 0
        STATE.current_epoch = 0
        STATE.total_epochs = epochs  
        STATE.last_error = None
        STATE.run_name = run_name
        STATE.best_pt_path = None
        STATE.exported_model_path = None

    push_log("Starting training...")
    push_log(f"data={data_arg}")
    push_log(f"model={model}, epochs={epochs}, imgsz={imgsz}, name={run_name}")
    if project_dir:
        push_log(f"project={project_dir}")

    cmd = [
        "yolo", "detect", "train",
        f"data={data_arg}",
        f"model={model}",
        f"epochs={epochs}",
        f"imgsz={imgsz}",
        f"name={run_name}",
        "exist_ok=True",
    ]
    if project_dir:
        cmd.append(f"project={project_dir}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            push_log(line)

    
            m = EPOCH_RE.search(line)
            if m:
                cur = int(m.group(1))
                with LOCK:
                    STATE.current_epoch = cur
                    STATE.progress_pct = int((cur / max(STATE.total_epochs, 1)) * 100)
                continue

    
            m2 = EPOCH_LINESTART_RE.match(line)
            if m2:
                cur = int(m2.group(1))
                total = int(m2.group(2))

        
                with LOCK:
                    expected = STATE.total_epochs

                if total == expected:
                    with LOCK:
                        STATE.current_epoch = cur
                        STATE.progress_pct = int((cur / max(expected, 1)) * 100)
                    continue

        # Training succeeded: locate best.pt
        proj = project_dir if project_dir else os.path.join(os.getcwd(), "runs", "detect")
        best_pt = os.path.join(proj, run_name, "weights", "best.pt")

        push_log(f"Looking for weights: {best_pt}")

        if not os.path.isfile(best_pt):
            with LOCK:
                STATE.status = "error"
                STATE.last_error = "Training finished but best.pt was not found. Check run output path."
            push_log(STATE.last_error)
            return

        with LOCK:
            STATE.status = "done"
            STATE.progress_pct = 100
            STATE.current_epoch = STATE.total_epochs  
            STATE.best_pt_path = best_pt
            STATE.exported_model_path = None

        push_log("Training finished successfully.")
        push_log("Click 'Download .pt' to choose where to save the trained weights.")

    except FileNotFoundError:
        with LOCK:
            STATE.status = "error"
            STATE.last_error = "Could not find 'yolo' command. Install ultralytics and ensure it's on PATH."
        push_log(STATE.last_error)

    except Exception as e:
        with LOCK:
            STATE.status = "error"
            STATE.last_error = str(e)
        push_log(f"Error: {e}")


# Routes
@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/state")
def api_state():
    with LOCK:
        return jsonify({
            "dataset_path": STATE.dataset_path,
            "output_path": STATE.output_path,
            "status": STATE.status,
            "progress_pct": STATE.progress_pct,
            "current_epoch": STATE.current_epoch,
            "total_epochs": STATE.total_epochs,
            "last_error": STATE.last_error,
            "run_name": STATE.run_name,
            "best_pt_path": STATE.best_pt_path,
            "exported_model_path": STATE.exported_model_path,
        })

@app.get("/api/runs")
def api_runs():
    base = _runs_base_dir()

    if not os.path.isdir(base):
        return jsonify({"ok": True, "base_dir": base, "runs": []})

    items = []
    for name in os.listdir(base):
        p = os.path.join(base, name)
        if os.path.isdir(p):
            try:
                mtime = os.path.getmtime(p)
            except Exception:
                mtime = 0
            items.append((name, mtime))

    
    items.sort(key=lambda x: x[1], reverse=True)
    runs = [name for name, _ in items]

    return jsonify({"ok": True, "base_dir": base, "runs": runs})

@app.post("/api/pick_dataset")
def api_pick_dataset():
    path = pick_directory_dialog("Select YOLO Dataset Folder")
    if not path:
        return jsonify({"ok": False, "error": "No folder selected (dialog canceled)."}), 400

    ok, msg = validate_yolo_dataset(path)
    if not ok:
        return jsonify({"ok": False, "error": msg, "path": path}), 400

    with LOCK:
        STATE.dataset_path = path

    push_log(f"Dataset selected: {path}")
    return jsonify({"ok": True, "path": path})


@app.post("/api/pick_output")
def api_pick_output():
    path = pick_directory_dialog("Select Output Folder (optional)")
    if not path:
        return jsonify({"ok": False, "error": "No folder selected (dialog canceled)."}), 400

    with LOCK:
        STATE.output_path = path

    push_log(f"Output folder selected: {path}")
    return jsonify({"ok": True, "path": path})


@app.post("/api/start_train")
def api_start_train():
    data = request.get_json(force=True) or {}

    epochs = int(data.get("epochs", 50))
    imgsz = int(data.get("imgsz", 640))
    model = str(data.get("model", "yolov8n.pt"))

    dataset_mode = data.get("dataset_mode", "local")
    preset_yaml = str(data.get("preset_yaml", "coco8.yaml"))
    run_name = _safe_run_name(str(data.get("run_name", "") or ""))

    with LOCK:
        running = (STATE.status == "running")
        ds = STATE.dataset_path
        out = STATE.output_path

    if running:
        return jsonify({"ok": False, "error": "Training is already running."}), 409

    
    if dataset_mode == "preset":
        data_arg = preset_yaml
    else:
        if not ds:
            return jsonify({"ok": False, "error": "No dataset selected."}), 400
        data_yaml = os.path.join(ds, "data.yaml")
        data_arg = data_yaml if os.path.isfile(data_yaml) else ds

    
    with LOCK:
        STATE.logs = queue.Queue()
        STATE.last_error = None
        STATE.best_pt_path = None
        STATE.exported_model_path = None

    t = threading.Thread(
        target=run_training_job,
        args=(data_arg, epochs, imgsz, model, out, run_name),
        daemon=True
    )
    t.start()

    return jsonify({"ok": True, "run_name": run_name})


@app.post("/api/export_model")
def api_export_model():
    
    with LOCK:
        best_pt = STATE.best_pt_path
        run_name = STATE.run_name or "trained_model"

    if not best_pt or not os.path.isfile(best_pt):
        return jsonify({"ok": False, "error": "No trained model found yet. Train a model first."}), 404

    export_path = pick_save_file_dialog("Save trained model as...", f"{run_name}.pt")
    if not export_path:
        return jsonify({"ok": False, "error": "Save canceled."}), 400

    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    shutil.copy2(best_pt, export_path)

    with LOCK:
        STATE.exported_model_path = export_path

    push_log(f"Saved model to: {export_path}")
    return jsonify({"ok": True, "exported_model_path": export_path})


@app.get("/api/logs")
def api_logs():
    
    def event_stream():
        last_ping = time.time()
        while True:
            try:
                line = STATE.logs.get(timeout=0.5)
                yield f"event: log\ndata: {json.dumps({'line': line})}\n\n"
            except queue.Empty:
                pass

            now = time.time()
            if now - last_ping >= 1.0:
                with LOCK:
                    payload = {
                        "status": STATE.status,
                        "progress_pct": STATE.progress_pct,
                        "current_epoch": STATE.current_epoch,
                        "total_epochs": STATE.total_epochs,
                        "last_error": STATE.last_error,
                        "best_pt_path": STATE.best_pt_path,
                        "exported_model_path": STATE.exported_model_path,
                    }
                yield f"event: progress\ndata: {json.dumps(payload)}\n\n"
                last_ping = now

    return Response(event_stream(), mimetype="text/event-stream")

@app.get("/metrics")
def metrics_page():
    # default to current STATE.run_name 
    with LOCK:
        run_name = STATE.run_name
    return render_template("metrics.html", run_name=run_name or "")

@app.get("/api/metrics")
def api_metrics():
    run_name = request.args.get("run", "").strip()
    if not run_name:
        with LOCK:
            run_name = STATE.run_name or ""

    if not run_name:
        return jsonify({"ok": False, "error": "No run available yet."}), 404

    run_dir = _run_dir_for(run_name)
    try:
        run_dir = _safe_join(run_dir)  
    except ValueError:
        return jsonify({"ok": False, "error": "Invalid run path."}), 400

    csv_path = os.path.join(run_dir, "results.csv")
    metrics = _read_last_metrics_csv(csv_path)

    
    images = []
    for fn in [
        "results.png",
        "confusion_matrix.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
        "labels.jpg",
        "labels_correlogram.jpg",
    ]:
        if os.path.isfile(os.path.join(run_dir, fn)):
            images.append(fn)

    return jsonify({
        "ok": True,
        "run_name": run_name,
        "run_dir": run_dir,
        "metrics": metrics,
        "images": images,
    })

@app.get("/runs/<run_name>/<path:filename>")
def serve_run_file(run_name: str, filename: str):
    run_dir = _run_dir_for(run_name)
    try:
        run_dir = _safe_join(run_dir)
        file_path = _safe_join(run_dir, filename)
    except ValueError:
        return "Invalid path", 400

    if not os.path.isfile(file_path):
        return "Not found", 404

    
    return send_from_directory(run_dir, filename)

if __name__ == "__main__":
   
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
