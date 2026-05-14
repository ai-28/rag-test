from __future__ import annotations

import asyncio
import json

from rag_api.service import ingest_pdf


def main() -> None:
    result = asyncio.run(ingest_pdf())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
