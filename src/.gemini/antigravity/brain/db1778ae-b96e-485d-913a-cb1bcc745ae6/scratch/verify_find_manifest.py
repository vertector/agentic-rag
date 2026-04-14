
import asyncio
import json
from pathlib import Path
from src.ingestion_pipeline.server import find_manifest

class MockContext:
    pass

async def test():
    ctx = MockContext()
    # Assuming bbs.pdf was parsed and exists in .cache/snapshots
    result = await find_manifest(ctx, "bbs.pdf")
    print(result)

if __name__ == "__main__":
    asyncio.run(test())
