"""QR codes for the footer (paper DOI + code repository)."""
from pathlib import Path
import qrcode
OUT = Path(__file__).resolve().parents[1] / "figures"
for name, url in {"qr_paper": "https://doi.org/10.1109/CODASSCA69992.2026.00059",
                  "qr_code": "https://github.com/Nard248/spectral-select"}.items():
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=12, border=1)
    qr.add_data(url); qr.make(fit=True)
    img = qr.make_image(fill_color="#1a2332", back_color="white")
    img.save(OUT / f"{name}.png"); print("wrote", name, img.size)
