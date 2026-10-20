# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Placeholders for internal features that the public Open-H pipeline does not ship.

Any access raises, so a configuration that would need one of them fails loudly instead of
silently changing behaviour.
"""


def __getattr__(name: str):
    raise NotImplementedError(
        f"{name!r} belongs to a feature (latent cache / multiview cross-view) that is not part of the "
        "public Open-H pipeline."
    )
