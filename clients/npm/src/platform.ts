// -----------------------------------------------------------------------------
//  Copyright (c) NoMercy Entertainment
//
//  Licensed under the MIT License. See LICENSE for details.
//
//  SPDX-License-Identifier: MIT
// -----------------------------------------------------------------------------

import { arch as osArch, platform as osPlatform } from 'node:os';

/** A release artifact target: the slug + archive extension the fork publishes. */
export interface PlatformTarget {
	/** Release-asset slug, e.g. `windows-x86_64`. */
	slug: string;
	/** Archive extension, e.g. `zip` or `tar.gz`. */
	ext: 'zip' | 'tar.gz';
	/** ffmpeg binary filename inside the archive. */
	ffmpeg: string;
	/** ffprobe binary filename inside the archive. */
	ffprobe: string;
}

const TARGETS: Record<string, PlatformTarget> = {
	'win32-x64': { slug: 'windows-x86_64', ext: 'zip', ffmpeg: 'ffmpeg.exe', ffprobe: 'ffprobe.exe' },
	'win32-arm64': { slug: 'windows-aarch64', ext: 'zip', ffmpeg: 'ffmpeg.exe', ffprobe: 'ffprobe.exe' },
	'linux-x64': { slug: 'linux-x86_64', ext: 'tar.gz', ffmpeg: 'ffmpeg', ffprobe: 'ffprobe' },
	'linux-arm64': { slug: 'linux-aarch64', ext: 'tar.gz', ffmpeg: 'ffmpeg', ffprobe: 'ffprobe' },
	'darwin-arm64': { slug: 'darwin-arm64', ext: 'tar.gz', ffmpeg: 'ffmpeg', ffprobe: 'ffprobe' },
	'darwin-x64': { slug: 'darwin-x86_64', ext: 'tar.gz', ffmpeg: 'ffmpeg', ffprobe: 'ffprobe' },
};

/**
 * Resolves the current OS+arch (or an explicit pair) to the release artifact the
 * fork publishes. Throws with an actionable message on an unsupported platform.
 */
export function resolvePlatformTarget(platform: string = osPlatform(), arch: string = osArch()): PlatformTarget {
	const key = `${platform}-${arch}`;
	const target = TARGETS[key];
	if (!target) {
		const supported = Object.keys(TARGETS).join(', ');
		throw new Error(
			`No NoMercy ffmpeg artifact for ${key}. Supported: ${supported}. `
			+ `Set NOMERCY_FFMPEG to a pre-supplied binary path to bypass platform detection.`,
		);
	}
	return target;
}
