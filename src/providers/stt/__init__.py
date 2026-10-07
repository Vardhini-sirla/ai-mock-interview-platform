"""Concrete speech-to-text implementations.

Nothing is re-exported here on purpose. Importing the provider classes at
package level would defeat the lazy-import contract described in
:mod:`src.providers`: a user with only one SDK installed must still be able to
import this package. Construct providers through
:func:`src.providers.get_stt` instead of importing from these modules directly.
"""

from __future__ import annotations
