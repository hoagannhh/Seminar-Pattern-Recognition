# ============================================================
#  setup_colab.py  —  Seminar-Pattern-Recognition
#  Chạy cell này ĐẦU TIÊN mỗi khi mở Colab session mới
#
#  Luồng hoạt động:
#    1. Clone / pull repo từ GitHub
#    2. Mount Google Drive (tuỳ chọn nhưng khuyến nghị)
#    3. Sync data/ models/ reports/ từ Drive → /content/
#    4. Cài thư viện
#  Để lưu kết quả về Drive sau mỗi notebook, gọi:
#    from setup_colab import sync_to_drive
#    sync_to_drive()
# ============================================================

import os
import sys
import shutil
import subprocess
from pathlib import Path

# ── Cấu hình ────────────────────────────────────────────────
GITHUB_USERNAME = "hoagannhh"
REPO_NAME       = "Seminar-Pattern-Recognition"
BRANCH          = "main"

# Tên thư mục trong Google Drive của bạn
DRIVE_FOLDER    = "Seminar-Pattern-Recognition"

PROJECT_ROOT = Path(f"/content/{REPO_NAME}")
DRIVE_ROOT   = Path(f"/content/drive/MyDrive/{DRIVE_FOLDER}")

# ── 1. GitHub Token ──────────────────────────────────────────
try:
    from google.colab import userdata
    GITHUB_TOKEN = userdata.get("GITHUB_TOKEN")
    print("🔑 Token lấy từ Colab Secrets")
except Exception:
    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

if not GITHUB_TOKEN:
    import getpass
    GITHUB_TOKEN = getpass.getpass("🔑 Nhập GitHub Personal Access Token: ")

# ── 2. Clone hoặc pull repo ──────────────────────────────────
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
    remote_url = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
    subprocess.run(["git", "-C", str(PROJECT_ROOT), "remote", "set-url", "origin", remote_url],
                   capture_output=True)
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "pull", "origin", BRANCH],
        capture_output=True, text=True
    )
    print("✅ Pull xong" if result.returncode == 0 else f"⚠️ Pull lỗi: {result.stderr}")

# ── 3. Set working directory ─────────────────────────────────
os.chdir(PROJECT_ROOT)
print(f"📁 Working dir: {os.getcwd()}")

# ── 4. Thêm src/ vào Python path ────────────────────────────
src_path = str(PROJECT_ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
print("🐍 src/ đã thêm vào sys.path")

# ── 5. Tạo thư mục cần thiết ────────────────────────────────
for folder in ["data/raw", "data/processed", "data/features", "models", "reports"]:
    (PROJECT_ROOT / folder).mkdir(parents=True, exist_ok=True)
print("📂 Thư mục data/ models/ reports/ sẵn sàng")

# ── 6. Mount Google Drive (persistent storage) ───────────────
_DRIVE_MOUNTED = False
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    _DRIVE_MOUNTED = True
    print("✅ Google Drive đã mount tại /content/drive/MyDrive/")

    # Tạo thư mục dự án trên Drive nếu chưa có
    for folder in ["data/raw", "data/processed", "data/features", "models", "reports"]:
        (DRIVE_ROOT / folder).mkdir(parents=True, exist_ok=True)
    print(f"📂 Drive folder: {DRIVE_ROOT}")
except Exception as e:
    print(f"⚠️  Không mount được Drive: {e}")
    print("   → File sẽ mất khi runtime reset. Khuyến nghị mount Drive để lưu lâu dài.")

# ── 7. Sync từ Drive → /content/ (nếu Drive mounted) ────────
_SYNC_FOLDERS = ["data/raw", "data/processed", "data/features", "models", "reports"]

def _copy_newer(src: Path, dst: Path) -> int:
    """Copy files từ src/ sang dst/ nếu chưa có hoặc Drive mới hơn. Trả về số file đã copy."""
    if not src.exists():
        return 0
    count = 0
    for src_file in src.rglob("*"):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(src)
        dst_file = dst / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        if not dst_file.exists() or src_file.stat().st_mtime > dst_file.stat().st_mtime:
            shutil.copy2(src_file, dst_file)
            count += 1
    return count


if _DRIVE_MOUNTED:
    print("\n🔄 Sync Drive → /content/ ...")
    total = 0
    for folder in _SYNC_FOLDERS:
        n = _copy_newer(DRIVE_ROOT / folder, PROJECT_ROOT / folder)
        if n:
            print(f"   {folder}: {n} file(s) copied")
        total += n
    print(f"✅ Sync xong ({total} file(s) total)")
else:
    # Kiểm tra data/raw có file chưa
    raw_files = list((PROJECT_ROOT / "data" / "raw").glob("*.csv"))
    if not raw_files:
        print("\n⚠️  data/raw/ trống — bạn cần:")
        print("   Option A: Mount Drive (chạy lại setup) rồi để script tự sync")
        print("   Option B: Chạy notebook 01_data_collection.ipynb để fetch từ SPARQL")
        print("   Option C: Upload file thủ công:")
        print("             from google.colab import files")
        print("             files.upload()  # chọn wikicities_raw.csv")
        print("             import shutil")
        print("             shutil.move('wikicities_raw.csv', 'data/raw/wikicities_raw.csv')")

# ── 8. Cài thư viện ──────────────────────────────────────────
req = PROJECT_ROOT / "requirements.txt"
if req.exists():
    print("\n📦 Cài thư viện...")
    subprocess.run(["pip", "install", "-r", str(req), "-q"])
    print("✅ Cài xong")

# ── 9. Kiểm tra GPU ──────────────────────────────────────────
gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    capture_output=True, text=True
)
print(f"🖥️  GPU: {gpu.stdout.strip()}" if gpu.returncode == 0
      else "⚠️  Không thấy GPU — vào Runtime > Change runtime type > GPU")

print("\n🚀 Setup hoàn tất! Bắt đầu chạy notebook.")


# ── Helper: sync ngược về Drive ─────────────────────────────
def sync_to_drive(verbose: bool = True) -> None:
    """
    Copy toàn bộ data/, models/, reports/ từ /content/ về Google Drive.
    Gọi hàm này ở cuối mỗi notebook để lưu kết quả lâu dài.

    Example
    -------
    >>> from setup_colab import sync_to_drive
    >>> sync_to_drive()
    """
    if not _DRIVE_MOUNTED:
        print("⚠️  Drive chưa mount — không thể sync. Chạy lại setup_colab.py.")
        return

    if verbose:
        print("💾 Sync /content/ → Drive ...")
    total = 0
    for folder in _SYNC_FOLDERS:
        n = _copy_newer(PROJECT_ROOT / folder, DRIVE_ROOT / folder)
        if verbose and n:
            print(f"   {folder}: {n} file(s) saved to Drive")
        total += n
    if verbose:
        print(f"✅ Đã lưu {total} file(s) về {DRIVE_ROOT}")