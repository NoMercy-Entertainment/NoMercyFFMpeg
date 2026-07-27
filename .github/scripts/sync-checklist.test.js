const assert = require('assert');
const { assess, assessHardware, rewriteChecklist } = require('./sync-checklist');

const HEAD = 'a'.repeat(40);
const CONTEXT = { headSha: HEAD, tag: 'v1.0.39-rc' };

const passingVerdict = (overrides = {}) => ({
  platform: 'linux-x86_64',
  tag: 'v1.0.39-rc',
  commit: HEAD,
  verdict: 'pass',
  integrity: { verified: true, sha256: 'f'.repeat(64) },
  report: {
    totals: { passed: 22, failed: 0, skipped: 4, total: 26 },
    tests: [{ name: 'LIBX264', status: 'passed' }],
  },
  ...overrides,
});

// ── A tick requires proof, and each missing proof is its own refusal ───────
assert.strictEqual(assess(passingVerdict(), CONTEXT).ok, true, 'a complete passing verdict ticks');

assert.strictEqual(assess(undefined, CONTEXT).ok, false, 'a missing verdict never ticks');
assert.match(assess(undefined, CONTEXT).why, /no verdict/, 'says the verdict is missing');

assert.strictEqual(
  assess(passingVerdict({ commit: 'b'.repeat(40) }), CONTEXT).ok,
  false,
  'a verdict for another commit never ticks',
);

assert.strictEqual(
  assess(passingVerdict({ tag: 'v1.0.38-rc' }), CONTEXT).ok,
  false,
  'a verdict for another RC never ticks',
);

assert.strictEqual(
  assess(passingVerdict({ integrity: { verified: false } }), CONTEXT).ok,
  false,
  'an unchecked artifact digest never ticks',
);

assert.strictEqual(
  assess(passingVerdict({ report: { totals: { passed: 0, failed: 0, skipped: 0, total: 0 }, tests: [] } }), CONTEXT).ok,
  false,
  'an empty result set never ticks',
);

// A suite can report pass while carrying failures; the totals are authoritative.
assert.strictEqual(
  assess(passingVerdict({
    report: {
      totals: { passed: 20, failed: 2, skipped: 4, total: 26 },
      tests: [{ name: 'LIBX264', status: 'failed' }],
    },
  }), CONTEXT).ok,
  false,
  'a verdict claiming pass with failed totals never ticks',
);

assert.strictEqual(
  assess(passingVerdict({ verdict: 'fail', report: { totals: { passed: 1, failed: 1, total: 2, skipped: 0 }, tests: [{ name: 'NVENC', status: 'failed' }] } }), CONTEXT).ok,
  false,
  'an explicit fail never ticks',
);

// ── Hardware line ─────────────────────────────────────────────────────────
const withTests = (platform, tests) =>
  new Map([[platform, { platform, report: { tests } }]]);

let hw = assessHardware(withTests('windows-x86_64', [
  { name: 'NVENC', status: 'passed' },
  { name: 'AMF', status: 'skipped', reason: 'no AMD display adapter' },
]));
assert.strictEqual(hw.ok, true, 'one real accelerator satisfies the hardware line');
assert.deepStrictEqual(hw.exercised, ['nvenc on windows-x86_64'], 'names what actually ran');
assert.strictEqual(hw.unavailable.length, 1, 'records what could not be tested');

hw = assessHardware(withTests('linux-x86_64', [
  { name: 'NVENC', status: 'skipped', reason: 'no NVIDIA adapter' },
  { name: 'AMF', status: 'skipped', reason: 'no AMD adapter' },
]));
assert.strictEqual(hw.ok, false, 'all-skipped acceleration does not satisfy the line');

hw = assessHardware(withTests('windows-x86_64', [{ name: 'NVENC', status: 'failed' }]));
assert.strictEqual(hw.ok, false, 'a failed accelerator fails the line');

// ── Rewriting is confined to the delimited section ────────────────────────
const body = [
  'intro',
  '- [ ] a stray box above the markers',
  '<!-- TEST-CHECKLIST-START -->',
  '- [ ] linux-x86_64 — tested on real hardware',
  '- [x] darwin-arm64 — tested on real hardware',
  '<!-- TEST-CHECKLIST-END -->',
  '- [ ] a stray box below the markers',
].join('\n');

const rewritten = rewriteChecklist(body, (label) => label.startsWith('linux-x86_64'));
assert.match(rewritten, /- \[x\] linux-x86_64/, 'ticks a verified platform');
assert.match(rewritten, /- \[ \] darwin-arm64/, 'unticks a platform that lost its proof');
assert.match(rewritten, /- \[ \] a stray box above/, 'leaves boxes above the markers alone');
assert.match(rewritten, /- \[ \] a stray box below/, 'leaves boxes below the markers alone');

assert.strictEqual(rewriteChecklist('no markers', () => true), null, 'refuses a body with no delimiters');

console.log('all sync-checklist tests passed');
