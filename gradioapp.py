import gradio as gr
import subprocess
import re

def imagegen_pipeline(image_path, optional_tags):
    logs = []
    
    # Run unified pipeline
    cmd = ["bash", "pipeline.sh", image_path]
    if optional_tags and optional_tags.strip():
        cmd += ["-t", optional_tags]
    
    proc = subprocess.run(cmd, capture_output=True, text=True)
    logs.append("[Pipeline stdout]\n" + (proc.stdout or "").strip())
    if proc.stderr:
        logs.append("[Pipeline stderr]\n" + proc.stderr.strip())
    
    if proc.returncode != 0:
        return None, "\n\n".join(logs)
    
    # Look for the output image path in stdout
    stdout = proc.stdout or ""
    saved_match = re.search(r"^Image saved as\s*(.+)$", stdout, re.MULTILINE)
    if not saved_match:
        logs.append("[App] Could not find 'Image saved as ...' in pipeline output.")
        return None, "\n\n".join(logs)
    
    output_path = saved_match.group(1).strip()
    logs.append(f"[App] Output image: {output_path}")
    return output_path, "\n\n".join(logs)

demo = gr.Interface(
    fn=imagegen_pipeline,
    inputs=[gr.Image(label="Input Image", type="filepath"), gr.Textbox(label="Optional Tags", value="")],
    outputs=[gr.Image(label="Output Image", type="filepath"), gr.Textbox(label="Logs")],
    title="GenshinfyV2 !!",
    description="Generate an avatar-style image of your face from a Genshin Impact character reference.",
    theme="default"
)

if __name__ == "__main__":
    demo.launch(pwa=True)