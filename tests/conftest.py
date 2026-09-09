"""Keep unit tests offline; subprocess CLI tests explicitly request --offline."""

import socket
from typing import NoReturn

import pytest


@pytest.fixture(autouse=True)
def block_unit_test_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail unexpected socket connections; this fixture has no financial units."""

    def forbidden(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("Unit tests must not access the network")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
