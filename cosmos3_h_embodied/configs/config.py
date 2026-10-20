# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Config entry point: cosmos-framework's base config plus this package's experiments.

Use it wherever cosmos-framework takes ``--config-file``, e.g.::

    python -m cosmos_framework.scripts.export_config -o h16.yaml \\
        --config-file cosmos3_h_embodied/configs/config.py --experiment openh_h16_task_prompt
"""

from cosmos_framework.configs.base.config import make_config as _make_cf_config


def make_config():
    config = _make_cf_config()
    import cosmos3_h_embodied.configs.experiment_h16  # noqa: F401  (registers the experiments)

    return config
