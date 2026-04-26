from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import DATA_DIR, EMBEDDING_PROVIDER
from vectorstore.build_index import get_index_paths, get_manifest_path, rebuild_vector_store


def main() -> None:
    data_dir = str(DATA_DIR)
    store = rebuild_vector_store(data_dir=data_dir, persist=True)
    index_path, metadata_path = get_index_paths(data_dir)
    manifest_path = get_manifest_path(data_dir)

    print("Index rebuild complete")
    print(f"Data directory: {Path(data_dir)}")
    print(f"Embedding provider: {EMBEDDING_PROVIDER}")
    print(f"Indexed chunks: {len(store.documents)}")
    print(f"FAISS index: {index_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()