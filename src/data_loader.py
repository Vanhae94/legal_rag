import glob
from pathlib import Path
from typing import List, Dict, Any
from . import config

def read_docs() -> List[Dict[str, Any]]:
    docs = []
    for p in glob.glob(str(config.DATA_DIR / "**/*.txt"), recursive=True):
        t = Path(p).read_text(encoding="utf-8", errors="ignore").strip()
        if t: docs.append({"title": Path(p).stem, "text": t, "meta": {"source": p}})
    return docs

def chunk(text: str, size=config.CHUNK_SIZE, overlap=config.OVERLAP) -> List[str]:
    text = text.replace("\r\n","\n").replace("\r","\n")
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out = []
    for para in paras:
        if len(para) <= size: out.append(para); continue
        s=0
        while s < len(para):
            e=s+size; out.append(para[s:e]); s=e-overlap
            if s<0: s=0
            if s>=len(para): break
    return out

