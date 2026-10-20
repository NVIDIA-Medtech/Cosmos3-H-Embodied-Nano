# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Test session setup: script entry points are imported for their functions only."""

import cosmos_framework.inference.common.init as _cf_init

# ``init_script`` (called at import time by script modules) refuses to run once other
# modules are loaded, which is always the case inside a pytest session.
_cf_init.init_script = _cf_init._init_script = lambda *args, **kwargs: None
