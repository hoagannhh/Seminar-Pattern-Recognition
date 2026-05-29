# ============================================================
#  setup_colab.py  —  Seminar-Pattern-Recognition
#  Chạy cell này ĐẦU TIÊN mỗi khi mở Colab session mới
# ============================================================

import sys
import subprocess
from pathlib import Path

# ── Cấu hình ────────────────────────────────────────────────
GITHUB_USERNAME = "hoagannhh"
REPO_NAME       = "Seminar-Pattern-Recognition"
BRANCH          = "main"
GITHUB_TOKEN    = ""  
# ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(f"/content/{REPO_NAME}")

# ── 1. Clone hoặc pull repo ──────────────────────────────────
if not PROJECT_ROOT.exists():
    print("📥 Cloning repo...")
    result = subprocess.run(
        ["git", "clone", "-b", BRANCH,
         f"https://{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git",
         str(PROJECT_ROOT)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"❌ Clone thất bại:\n{result.stderr}")
    print("✅ Clone xong")
else:
    print("🔄 Repo đã có, pulling latest...")
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "pull", "origin", BRANCH],
        capture_output=True, text=True
    )
    print("✅ Pull xong" if result.returncode == 0 else f"⚠️ Pull lỗi: {result.stderr}")

# ── 2. Set working directory ─────────────────────────────────
import os
os.chdir(PROJECT_ROOT)
print(f"📁 Working dir: {os.getcwd()}")

# ── 3. Thêm src/ vào Python path ────────────────────────────
src_path = str(PROJECT_ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
print("🐍 src/ đã thêm vào sys.path")

# ── 4. Cài thư viện ──────────────────────────────────────────
req = PROJECT_ROOT / "requirements.txt"
if req.exists():
    print("📦 Cài thư viện...")
    subprocess.run(["pip", "install", "-r", str(req), "-q"])
    print("✅ Cài xong")

# ── 5. Tạo thư mục cần thiết ────────────────────────────────
for folder in ["data/raw", "data/processed", "data/features", "models", "reports"]:
    (PROJECT_ROOT / folder).mkdir(parents=True, exist_ok=True)
print("📂 Thư mục data/ models/ reports/ sẵn sàng")

# ── 6. Kiểm tra GPU ──────────────────────────────────────────
gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    capture_output=True, text=True
)
print(f"🖥️  GPU: {gpu.stdout.strip()}" if gpu.returncode == 0
      else "⚠️  Không thấy GPU — vào Runtime > Change runtime type > GPU")

print("\n🚀 Setup hoàn tất! Bắt đầu chạy notebook.")