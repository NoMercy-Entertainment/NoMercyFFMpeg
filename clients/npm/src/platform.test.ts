// -----------------------------------------------------------------------------
//  Copyright (c) NoMercy Entertainment
//
//  Licensed under the MIT License. See LICENSE for details.
//
//  SPDX-License-Identifier: MIT
// -----------------------------------------------------------------------------

import { describe, expect, it } from 'vitest';
import { resolvePlatformTarget } from './platform.js';
import { assetUrl } from './resolve.js';

describe('resolvePlatformTarget', () => {
	it('maps every supported OS+arch to the published artifact', () => {
		expect(resolvePlatformTarget('win32', 'x64')).toEqual({
			slug: 'windows-x86_64',
			ext: 'zip',
			ffmpeg: 'ffmpeg.exe',
			ffprobe: 'ffprobe.exe',
		});
		expect(resolvePlatformTarget('linux', 'x64')).toEqual({
			slug: 'linux-x86_64',
			ext: 'tar.gz',
			ffmpeg: 'ffmpeg',
			ffprobe: 'ffprobe',
		});
		expect(resolvePlatformTarget('linux', 'arm64')).toEqual({
			slug: 'linux-aarch64',
			ext: 'tar.gz',
			ffmpeg: 'ffmpeg',
			ffprobe: 'ffprobe',
		});
		expect(resolvePlatformTarget('darwin', 'arm64')).toEqual({
			slug: 'darwin-arm64',
			ext: 'tar.gz',
			ffmpeg: 'ffmpeg',
			ffprobe: 'ffprobe',
		});
		expect(resolvePlatformTarget('darwin', 'x64')).toEqual({
			slug: 'darwin-x86_64',
			ext: 'tar.gz',
			ffmpeg: 'ffmpeg',
			ffprobe: 'ffprobe',
		});
	});

	// win32-arm64 used to be the example here. It is a supported target now
	// (windows-aarch64), so the unsupported case needs an arch the fork really
	// does not build -- 32-bit Windows.
	it('throws an actionable error for an unsupported platform', () => {
		expect(() => resolvePlatformTarget('win32', 'ia32')).toThrow(/No NoMercy ffmpeg artifact for win32-ia32/);
		expect(() => resolvePlatformTarget('aix', 'ppc64')).toThrow(/Set NOMERCY_FFMPEG/);
	});
});

describe('assetUrl', () => {
	it('builds the windows release URL with the pinned fork + ffmpeg versions', () => {
		const target = resolvePlatformTarget('win32', 'x64');
		expect(assetUrl(target)).toBe(
			'https://github.com/NoMercy-Entertainment/nomercy-ffmpeg/releases/download/v1.0.38/ffmpeg-9.0-windows-x86_64-v1.0.38.zip',
		);
	});

	it('builds a tarball release URL for unix targets', () => {
		const target = resolvePlatformTarget('linux', 'arm64');
		expect(assetUrl(target)).toBe(
			'https://github.com/NoMercy-Entertainment/nomercy-ffmpeg/releases/download/v1.0.38/ffmpeg-9.0-linux-aarch64-v1.0.38.tar.gz',
		);
	});

	// Windows on ARM resolves to its own artifact, not the x64 one. Node reports
	// arch 'arm64' natively there; it only reports 'x64' when Node itself is the
	// emulated x64 build, in which case the x64 binaries are the correct answer.
	it('builds the windows-aarch64 release URL for Windows on ARM', () => {
		const target = resolvePlatformTarget('win32', 'arm64');
		expect(target.slug).toBe('windows-aarch64');
		expect(target.ffmpeg).toBe('ffmpeg.exe');
		expect(assetUrl(target)).toBe(
			'https://github.com/NoMercy-Entertainment/nomercy-ffmpeg/releases/download/v1.0.38/ffmpeg-9.0-windows-aarch64-v1.0.38.zip',
		);
	});
});
