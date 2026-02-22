"""Patch ddgs to use a fixed Chrome impersonation instead of random.

Fixes: "Impersonate 'chrome_127' does not exist, using 'random'"
Uses chrome_120 + windows (widely supported) so JARVIS gets consistent browser-like requests.
"""

import logging
import os

logger = logging.getLogger(__name__)
_patched = False

# primp 1.0.0 supports: chrome_144, chrome_145, edge_144, edge_145, safari_18.5, firefox_140, etc.
_DEFAULT_IMPERSONATE = "chrome_145"
_DEFAULT_IMPERSONATE_OS = "windows"


def apply_ddgs_patch(config: dict | None = None) -> None:
    """Patch ddgs HttpClient to use fixed Chrome impersonation instead of random."""
    global _patched
    if _patched:
        return

    cfg = config or {}
    ddgs_cfg = cfg.get("ddgs", {})
    impersonate = (
        os.environ.get("JARVIS_DDGS_IMPERSONATE")
        or ddgs_cfg.get("impersonate")
        or _DEFAULT_IMPERSONATE
    )
    impersonate_os = (
        os.environ.get("JARVIS_DDGS_IMPERSONATE_OS")
        or ddgs_cfg.get("impersonate_os")
        or _DEFAULT_IMPERSONATE_OS
    )

    try:
        import ddgs.http_client as hc
        import primp

        _orig_init = hc.HttpClient.__init__

        def _patched_init(
            self,
            proxy: str | None = None,
            timeout: int | None = 10,
            *,
            verify: bool | str = True,
        ) -> None:
            self.client = primp.Client(
                proxy=proxy,
                timeout=timeout,
                impersonate=impersonate,
                impersonate_os=impersonate_os,
                verify=verify if isinstance(verify, bool) else True,
                ca_cert_file=verify if isinstance(verify, str) else None,
            )

        hc.HttpClient.__init__ = _patched_init
        _patched = True
        logger.debug("ddgs patched: impersonate=%s, impersonate_os=%s", impersonate, impersonate_os)
    except Exception as e:
        logger.warning("Could not patch ddgs HttpClient: %s", e)
