#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Thu Jul 19 17:09:55 CEST 2012

import functools
import importlib
import logging
import unittest

logger = logging.getLogger("bob.bio.face")


def is_library_available(library):
    """Decorator to check if the mxnet is present, before running the test"""

    def _is_library_available(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            try:
                importlib.import_module(library)

                return function(*args, **kwargs)
            except ImportError as e:
                # unittest.SkipTest is compatible with both nose and pytest
                raise unittest.SkipTest(
                    f"Skipping test since `{library}` is not available: %s" % e
                )

        return wrapper

    return _is_library_available
