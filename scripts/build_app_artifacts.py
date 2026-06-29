from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    from WebApp import service

    data, elasticity, _, _ = service.load_data()

    service.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    data.to_pickle(service.APP_DATA_ARTIFACT_PATH, compression="gzip")
    elasticity.to_pickle(service.ELASTICITY_ARTIFACT_PATH, compression="gzip")

    print(f"Wrote {service.APP_DATA_ARTIFACT_PATH}")
    print(f"Wrote {service.ELASTICITY_ARTIFACT_PATH}")
    print(f"App data size: {service.APP_DATA_ARTIFACT_PATH.stat().st_size:,} bytes")
    print(f"Elasticity size: {service.ELASTICITY_ARTIFACT_PATH.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
