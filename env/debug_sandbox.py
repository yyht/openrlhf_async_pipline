

import asyncio
from microsandbox import PythonSandbox

async def main():
    async with PythonSandbox.create(name="test") as sb:
        exec = await sb.run("name = 'Python'")
        exec = await sb.run("print(f'Hello {name}!')")

    print(await exec.output()) # prints Hello Python!

asyncio.run(main())