# ============================================================
#  setup_colab.py  —  Seminar-Pattern-Recognition
#  Chạy cell này ĐẦU TIÊN mỗi khi mở Colab session mới
# ============================================================

import os
import sys
import subprocess
from pathlib import Path

# ── Cấu hình ────────────────────────────────────────────────
GITHUB_USERNAME = "hoagannhh"
REPO_NAME       = "Seminar-Pattern-Recognition"
BRANCH          = "main"

# ── Lấy token an toàn (KHÔNG hardcode token vào đây) ────────
# Ưu tiên 1: Colab Secrets (vào 🔑 bên trái, thêm key GITHUB_TOKEN)
try:
    from google.colab import userdata
    GITHUB_TOKEN = userdata.get("GITHUB_TOKEN")
    print("🔑 Token lấy từ Colab Secrets")
except Exception:
    # Ưu tiên 2: biến môi trường
    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

if not GITHUB_TOKEN:
    # Ưu tiên 3: nhập tay (ẩn input)
    import getpass
    GITHUB_TOKEN = getpass.getpass("🔑 Nhập GitHub Personal Access Token: ")
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
    # Cập nhật remote URL với token để tránh lỗi interactive password prompt
    remote_url = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
    subprocess.run(["git", "-C", str(PROJECT_ROOT), "remote", "set-url", "origin", remote_url],
                   capture_output=True)
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "pull", "origin", BRANCH],
        capture_output=True, text=True
    )
    print("✅ Pull xong" if result.returncode == 0 else f"⚠️ Pull lỗi: {result.stderr}")

# ── 2. Set working directory ─────────────────────────────────
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
