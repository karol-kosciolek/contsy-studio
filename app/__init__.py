# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Contsy Studio - lokalna aplikacja do generowania tresci AI (offline)."""

__title__ = "Contsy Studio"
__version__ = "0.2.0"
__license__ = "GPL-3.0-only"  # own code; third-party parts: THIRD_PARTY_NOTICES.md
__copyright__ = "© 2026 Karol Kosciolek"

# Certyfikaty z systemowego magazynu Windows: pobieranie dziala tez za antywirusem lub
# proxy skanujacym HTTPS (np. Norton), ktory podstawia certyfikat nieznany bibliotece requests.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass
