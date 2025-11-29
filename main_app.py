import subprocess
import psutil
import os
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
running = {"binary": None, "colour": None}


def start_process(name, script, port):
    if running[name] is not None:
        return

    running[name] = subprocess.Popen(
        ["python3", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )


def stop_process(name):
    if running[name] is not None:
        proc = running[name].pid
        try:
            p = psutil.Process(proc)
            p.terminate()
        except:
            pass
        running[name] = None


@app.route("/")
def home():
    return render_template("main_menu.html")


@app.route("/launch", methods=["POST"])
def launch():
    mode = request.form.get("mode")
    if mode == "binary":
        start_process("binary", "Binary.py", 5001)
    if mode == "colour":
        start_process("colour", "Colour.py", 5002)
    return jsonify({"status": "started"})


@app.route("/stop", methods=["POST"])
def stop():
    mode = request.form.get("mode")
    if mode == "binary":
        stop_process("binary")
    if mode == "colour":
        stop_process("colour")
    return jsonify({"status": "stopped"})


@app.route("/status")
def status():
    return jsonify({
        "binary": running["binary"] is not None,
        "colour": running["colour"] is not None
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7000)
