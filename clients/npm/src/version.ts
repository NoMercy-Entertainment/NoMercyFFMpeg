// -----------------------------------------------------------------------------
//  Copyright (c) NoMercy Entertainment
//
//  Licensed under the MIT License. See LICENSE for details.
//
//  SPDX-License-Identifier: MIT
// -----------------------------------------------------------------------------

/** GitHub repository that publishes the per-platform fork artifacts. */
export const REPO = 'NoMercy-Entertainment/nomercy-ffmpeg';

/**
 * Pinned fork release tag. Overridable via `NOMERCY_FFMPEG_VERSION` so a CI job
 * can bump the binary without a package release.
 */
export const FORK_VERSION = process.env['NOMERCY_FFMPEG_VERSION'] ?? 'v1.0.38';

/**
 * Upstream ffmpeg version baked into the pinned fork release. Part of the asset
 * filename, so overridable via `NOMERCY_FFMPEG_FFVERSION` alongside the tag.
 */
export const FFMPEG_VERSION = process.env['NOMERCY_FFMPEG_FFVERSION'] ?? '9.0';
