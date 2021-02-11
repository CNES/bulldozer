# Copyright 2021 PIERRE LASSALLE
# All rights reserved

import logging

from bulldozer import bulldozer

LOGGER = logging.getLogger(__name__)


def test_hello_world():
    output = bulldozer.hello_world()
    LOGGER.info(output)
    assert output == "hello world!"
