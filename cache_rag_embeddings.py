import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import open_clip


# Helpers
def _build_clip(device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    model.to(device)
    return model, tokenizer


def _embed_text(text: str, model, tokenizer, device: str) -> np.ndarray:
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten()


def _truncate(text: str, max_words: int = 50) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


# Main logic
def build_cache(rag_dir: Path, cache_path: Path, device: str, logger: logging.Logger) -> None:
    txt_files = sorted(rag_dir.glob("*.txt"))
    if not txt_files:
        logger.error(f"No .txt files found in {rag_dir}")
        sys.exit(1)

    logger.info(f"Found {len(txt_files)} RAG example file(s) in {rag_dir}")
    logger.info(f"Loading CLIP model on device: {device}")
    model, tokenizer = _build_clip(device)

    texts: list[str] = []
    embeddings: list[np.ndarray] = []
    filenames: list[str] = []

    start = time.perf_counter()
    for idx, path in enumerate(txt_files, 1):
        raw = path.read_text(encoding="utf-8").strip()
        truncated = _truncate(raw)
        emb = _embed_text(truncated, model, tokenizer, device)

        texts.append(raw)
        embeddings.append(emb)
        filenames.append(path.name)

        logger.info(f"  [{idx}/{len(txt_files)}] {path.name}")

    elapsed = time.perf_counter() - start
    logger.info(f"Embedded {len(txt_files)} examples in {elapsed:.2f}s")

    # Stack into (N, D) matrix and save alongside the raw texts and filenames
    emb_matrix = np.stack(embeddings, axis=0)  # shape: (N, embedding_dim)

    np.savez_compressed(
        cache_path,
        embeddings=emb_matrix,
        texts=np.array(texts, dtype=object),
        filenames=np.array(filenames, dtype=object),
    )
    logger.info(f"Cache saved to {cache_path}  (shape: {emb_matrix.shape})")

# Implement main function with arg parser including a help method
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute CLIP embeddings for RAG example .txt files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--rag-directory",
        required=True,
        type=Path,
        help="Directory containing RAG example .txt files",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=None,
        help="Output .npz path (default: <rag-directory>/embeddings.npz)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing cache file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device: 'cpu' or 'cuda' (default: auto-detect)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    logger = logging.getLogger(__name__)

    rag_dir: Path = args.rag_directory.resolve()
    if not rag_dir.is_dir():
        logger.error(f"RAG directory not found: {rag_dir}")
        sys.exit(1)

    cache_path: Path = (
        args.cache_file.resolve()
        if args.cache_file
        else rag_dir / "embeddings.npz"
    )

    if cache_path.exists() and not args.force:
        logger.warning(
            f"Cache already exists at {cache_path}. "
            "Use --force to overwrite."
        )
        sys.exit(0)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    build_cache(rag_dir, cache_path, device, logger)
    logger.info("Done.")


if __name__ == "__main__":
    main()