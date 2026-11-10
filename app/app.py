import glob, os, time
import gradio as gr
import numpy as np
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
M1 = YOLO(os.path.join(BASE, "model", "tahap1_panel_int8.onnx"), task="detect")
M2 = YOLO(os.path.join(BASE, "model", "tahap2_digit_int8.onnx"), task="detect")
CONF1, CONF2, MARGIN = 0.4384, 0.4174, 0.25
DIGIT = [str(i) for i in range(10)]

# Hugging Face free tier menjalankan Gradio di ZeroGPU yang mewajibkan satu fungsi @spaces.GPU.
# Fungsi ini sengaja tidak dipakai: inferensi tetap di CPU, sama seperti evaluasi.
try:
    import spaces
    @spaces.GPU
    def _penanda_zerogpu():
        return None
except ImportError:
    pass

def rakit(boxes, kelas=DIGIT):
    odo = [b for b in boxes if b[0] == "odometer"]
    if not odo:
        return None
    _, ox, oy, ow, oh, _ = max(odo, key=lambda b: b[5])
    x1, y1, x2, y2 = ox - ow / 2, oy - oh / 2, ox + ow / 2, oy + oh / 2
    sim = [b for b in boxes if b[0] in kelas and x1 <= b[1] <= x2 and y1 <= b[2] <= y2]
    if not sim:
        return ""
    sim.sort(key=lambda b: b[2])
    tinggi = float(np.median([b[4] for b in sim]))
    baris = [[sim[0]]]
    for b in sim[1:]:
        if abs(b[2] - np.mean([s[2] for s in baris[-1]])) > 0.5 * tinggi:
            baris.append([b])
        else:
            baris[-1].append(b)
    return " ".join("".join(s[0] for s in sorted(r, key=lambda s: s[1])) for r in baris)

def ke_boxes(r):
    b = r.boxes
    return [(r.names[int(c)], *xywh, float(s)) for c, xywh, s in zip(b.cls.tolist(), b.xywhn.tolist(), b.conf.tolist())]

def baca(img_rgb):
    if img_rgb is None:
        return None, None, "Silakan unggah foto odometer", ""
    bgr = np.ascontiguousarray(img_rgb[:, :, ::-1]); t0 = time.perf_counter()
    r1 = M1.predict(bgr, imgsz=640, conf=CONF1, device="cpu", verbose=False)[0]
    odo = [b for b in ke_boxes(r1) if b[0] == "odometer"]
    if not odo:
        return r1.plot()[:, :, ::-1], None, "Odometer tidak terdeteksi - gambar ditolak", ""
    H, W = bgr.shape[:2]
    _, xc, yc, w, h, _ = max(odo, key=lambda b: b[5])
    x1, y1 = int(max((xc - (0.5 + MARGIN) * w) * W, 0)), int(max((yc - (0.5 + MARGIN) * h) * H, 0))
    x2, y2 = int(min((xc + (0.5 + MARGIN) * w) * W, W)), int(min((yc + (0.5 + MARGIN) * h) * H, H))
    r2 = M2.predict(np.ascontiguousarray(bgr[y1:y2, x1:x2]), imgsz=640, conf=CONF2, device="cpu", verbose=False)[0]
    bx2 = ke_boxes(r2) + [("odometer", 0.5, 0.5, 1.0, 1.0, 1.0)]
    teks = rakit(bx2) or "Panel terdeteksi, tetapi angka tidak terbaca"
    info = f"panel conf {odo[0][5]:.2f} | {len(bx2) - 1} karakter | waktu {(time.perf_counter() - t0) * 1000:.0f} ms (CPU)"
    return r1.plot()[:, :, ::-1], r2.plot()[:, :, ::-1], teks, info

contoh = sorted(glob.glob(os.path.join(BASE, "examples", "*")))
demo = gr.Interface(
    fn=baca, inputs=gr.Image(type="numpy", label="Foto odometer"),
    outputs=[gr.Image(label="Tahap 1: deteksi panel"), gr.Image(label="Tahap 2: pembacaan digit"),
             gr.Textbox(label="Jarak tempuh terbaca"), gr.Textbox(label="Info")],
    examples=contoh or None, flagging_mode="never",
    title="Pembacaan Odometer Otomatis - CNN Dua Tahap (YOLO11s INT8 selektif)",
    description=f"Tahap 1 menemukan panel odometer (threshold {CONF1:.3f}), tahap 2 membaca digit pada potongan panel "
                f"(threshold {CONF2:.3f}). Inferensi di CPU. Foto tanpa panel odometer akan ditolak.")
if __name__ == "__main__":
    demo.launch()
