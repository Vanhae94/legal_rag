# change_to_txt.py
from pathlib import Path
import win32com.client as win32

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def hwp_to_txt(hwp_path: Path, out_dir: Path) -> Path:
    """
    HWP를 열어 TXT로 저장한 뒤 UTF-8로 재인코딩해서 반환.
    """
    out_tmp = out_dir / (hwp_path.stem + ".ansi.txt")  # 한글이 저장하는 기본 텍스트(대개 CP949)
    out_utf8 = out_dir / (hwp_path.stem + ".txt")

    hwp = win32.Dispatch("HWPFrame.HwpObject")
    # 보안 모듈 등록(경로 확인 대화 상자 방지)
    try:
        hwp.RegisterModule("FilePathCheckDLL", "SecurityModule")
    except Exception:
        pass

    try:
        # HWP Open 은 (파일경로, format, option) 3개의 매개변수 필요
        # format/option 은 기본값 "" 넣어주면 됩니다.
        hwp.Open(str(hwp_path), "", "")
        
        # 가장 간단한 저장 방법
        try:
            hwp.SaveAs(str(out_tmp), "TEXT")  # 텍스트로 저장
        except Exception:
            # 일부 버전 호환용(액션 호출 방식)
            hwp.HAction.GetDefault("FileSaveAs_S", hwp.HParameterSet.HFileOpenSave.HSet)
            hwp.HParameterSet.HFileOpenSave.Filename = str(out_tmp)
            hwp.HParameterSet.HFileOpenSave.Format = "TEXT"
            hwp.HAction.Execute("FileSaveAs_S", hwp.HParameterSet.HFileOpenSave.HSet)

    finally:
        try:
            hwp.Quit()  # 한글 종료
        except Exception:
            pass

    # TXT를 UTF-8로 재저장
    text = None
    for enc in ("utf-8", "cp949", "euc-kr"):
        try:
            text = out_tmp.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = out_tmp.read_text(encoding="cp949", errors="ignore")

    out_utf8.write_text(text, encoding="utf-8")
    try:
        out_tmp.unlink()
    except FileNotFoundError:
        pass

    return out_utf8

def main():
    hwp_files = sorted(p for p in ROOT.glob("*.hwp") if p.is_file())
    if not hwp_files:
        print("루트에 .hwp 파일이 없습니다.")
        return

    for hwp_path in hwp_files:
        out_path = hwp_to_txt(hwp_path, OUT_DIR)
        print(f"✔ {hwp_path.name} -> {out_path.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
