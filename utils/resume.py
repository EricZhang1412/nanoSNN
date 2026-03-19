from pathlib import Path


def resolve_resume_ckpt(resume, ckpt_dir):
    if resume is None:
        return None

    resume = str(resume).strip()
    if resume.lower() in ("none", "null", ""):
        return None

    if resume.lower() != "auto":
        ckpt_path = Path(resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {ckpt_path}")
        return str(ckpt_path)

    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return None

    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        return str(last_ckpt)

    all_ckpts = sorted(
        ckpt_dir.glob("*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return str(all_ckpts[0]) if all_ckpts else None
