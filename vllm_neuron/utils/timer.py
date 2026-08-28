# SPDX-License-Identifier: Apache-2.0
import time
from contextlib import contextmanager


@contextmanager
def timer():
    start = time.time()
    yield lambda: time.time() - start
