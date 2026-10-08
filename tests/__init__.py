"""测试包(有 __init__.py,`python -m unittest discover -s tests -t .` 才能递归发现)。"""

import logging

# IsolatedAsyncioTestCase 会开启 asyncio 调试模式,于是每条慢回调都会打一行
# "Executing <Task ...> took x seconds"。这对我们的断言毫无信息量,直接压掉。
logging.getLogger("asyncio").setLevel(logging.ERROR)
