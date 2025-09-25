import gradio as gr
import subprocess
import re

def imagegen_pipeline(image_path, optional_tags):
    logs = []
    # DINO step
    dino_proc = subprocess.run(["bash", "DINOv2/dino.sh", image_path], capture_output=True, text=True)
    logs.append("[DINO stdout]\n" + (dino_proc.stdout or "").strip())
    if dino_proc.stderr:
        logs.append("[DINO stderr]\n" + dino_proc.stderr.strip())
    if dino_proc.returncode != 0:
        return None, "\n\n".join(logs)

    stdout = (dino_proc.stdout or "").strip()
    if not stdout:
        logs.append("[App] DINO returned no output path.")
        return None, "\n\n".join(logs)
    # Use the last non-empty line as the selected path
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    best_match_path = lines[-1]
    logs.append(f"[App] Using best match path: {best_match_path}")

    # Style transfer step
    cmd = ["bash", "AnythingV4/styletransfer.sh", best_match_path]
    if optional_tags and optional_tags.strip():
        cmd += ["-t", optional_tags]
    st_proc = subprocess.run(cmd, capture_output=True, text=True)
    logs.append("[Style stdout]\n" + (st_proc.stdout or "").strip())
    if st_proc.stderr:
        logs.append("[Style stderr]\n" + st_proc.stderr.strip())
    if st_proc.returncode != 0:
        return None, "\n\n".join(logs)

    st_out = st_proc.stdout or ""
    saved_match = re.search(r"^Image saved as\s*(.+)$", st_out, re.MULTILINE)
    if not saved_match:
        logs.append("[App] Could not find 'Image saved as ...' in style output.")
        return None, "\n\n".join(logs)
    output_path = saved_match.group(1).strip()
    logs.append(f"[App] Output image: {output_path}")
    return output_path, "\n\n".join(logs)

demo = gr.Interface(
    fn=imagegen_pipeline,
    inputs=[gr.Image(label="Input Image", type="filepath"), gr.Textbox(label="Optional Tags", value="")],
    outputs=[gr.Image(label="Output Image", type="filepath"), gr.Textbox(label="Logs")],
    title="GenshinfyV2 - AI Style Transfer Project",
    description="Generate an avatar-style image of your face from a Genshin Impact character reference.",
    theme="default"
)

if __name__ == "__main__":
    demo.launch()