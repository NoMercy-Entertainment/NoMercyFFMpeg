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
// assessHardware takes the ASSESSED results, so these build that shape. A raw
// verdict map no longer satisfies it, which is what stops the permissive
// behaviour being written down here as the specification.
const accepted = (platform, tests) =>
  new Map([[platform, { verdict: { platform, report: { tests } }, assessment: { ok: true } }]]);

let hw = assessHardware(accepted('windows-x86_64', [
  { name: 'NVENC', status: 'passed' },
  { name: 'AMF', status: 'skipped', reason: 'no AMD display adapter' },
]));
assert.strictEqual(hw.ok, true, 'one real accelerator satisfies the hardware line');
assert.deepStrictEqual(hw.exercised, ['nvenc on windows-x86_64'], 'names what actually ran');
assert.strictEqual(hw.unavailable.length, 1, 'records what could not be tested');

hw = assessHardware(accepted('linux-x86_64', [
  { name: 'NVENC', status: 'skipped', reason: 'no NVIDIA adapter' },
  { name: 'AMF', status: 'skipped', reason: 'no AMD adapter' },
]));
assert.strictEqual(hw.ok, false, 'all-skipped acceleration does not satisfy the line');

hw = assessHardware(accepted('windows-x86_64', [{ name: 'NVENC', status: 'failed' }]));
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

// A human-ticked line the script does not recognise must survive. Clearing it
// blocks the merge in check-checklist.js and gets cleared again on every push,
// so nobody can tick their way out of it.
const humanBody = [
  '<!-- TEST-CHECKLIST-START -->',
  '- [ ] linux-x86_64 — tested on real hardware',
  '- [x] Release notes reviewed by a human',
  '<!-- TEST-CHECKLIST-END -->',
].join('\n');
const humanKept = rewriteChecklist(humanBody, (label, wasChecked) =>
  label.startsWith('linux-x86_64') ? true : wasChecked);
assert.match(humanKept, /- \[x\] Release notes reviewed by a human/,
  'a human-ticked unrecognised item stays ticked');
assert.match(humanKept, /- \[x\] linux-x86_64/, 'and the recognised one still ticks');

// ── Missing proof is not the same as matching proof ───────────────────────
const { commit: _c, ...noCommit } = passingVerdict();
assert.strictEqual(assess(noCommit, CONTEXT).ok, false,
  'a verdict with no commit never ticks');
assert.strictEqual(assess({ ...passingVerdict(), commit: '' }, CONTEXT).ok, false,
  'an empty commit is a refusal, not a skipped check');

const { tag: _t, ...noTag } = passingVerdict();
assert.strictEqual(assess(noTag, CONTEXT).ok, false,
  'a verdict with no tag never ticks');
assert.strictEqual(assess({ ...passingVerdict(), tag: '' }, CONTEXT).ok, false,
  'an empty tag is a refusal, not a skipped check');

// ── The hardware line inherits every refusal assess() makes ───────────────
// A verdict assess() rejected must not tick the hardware box by another route.
// The linkage itself, not just the empty case: a verdict carrying a PASSING
// accelerator that assess() refused must not satisfy the hardware line.
const rejected = { ...passingVerdict(), commit: 'deadbeef', platform: 'windows-x86_64' };
rejected.report = { ...rejected.report, tests: [{ name: 'NVENC', status: 'passed' }] };
const rejectedAssessment = assess(rejected, CONTEXT);
assert.strictEqual(rejectedAssessment.ok, false, 'stale commit is refused');

const mixed = new Map([
  ['windows-x86_64', { verdict: rejected, assessment: rejectedAssessment }],
]);
assert.strictEqual(assessHardware(mixed).ok, false,
  'a passing accelerator inside a REFUSED verdict does not tick the hardware line');
assert.deepStrictEqual(assessHardware(mixed).exercised, [],
  'and it is not even reported as exercised');

// The same verdict, accepted, does satisfy it - so the filter has not
// overshot into refusing everything.
const okAssessment = { ok: true };
const sameButAccepted = new Map([
  ['windows-x86_64', { verdict: rejected, assessment: okAssessment }],
]);
assert.strictEqual(assessHardware(sameButAccepted).ok, true,
  'the identical verdict, once accepted, satisfies the hardware line');

assert.strictEqual(assessHardware(new Map()).ok, false,
  'with no verdicts at all the hardware line cannot be satisfied');
console.log('all sync-checklist tests passed');

// ── Verdicts must be found however the artifacts were laid out ────────────
// download-artifact flattens; `gh run download` nests one folder per artifact.
// A non-recursive read of the nested layout finds nothing and would report every
// healthy platform as unverified.
const fs = require('fs');
const os = require('os');
const pathModule = require('path');
const { readVerdicts } = require('./sync-checklist');

const root = fs.mkdtempSync(pathModule.join(os.tmpdir(), 'verdicts-'));
fs.writeFileSync(pathModule.join(root, 'verdict-flat.json'), JSON.stringify({ platform: 'linux-x86_64' }));
const nested = pathModule.join(root, 'verdict-darwin-arm64');
fs.mkdirSync(nested);
fs.writeFileSync(pathModule.join(nested, 'verdict-darwin-arm64.json'), JSON.stringify({ platform: 'darwin-arm64' }));

const discovered = readVerdicts(root);
assert.strictEqual(discovered.size, 2, 'finds both flat and nested verdicts');
assert.ok(discovered.has('linux-x86_64'), 'finds the flat verdict');
assert.ok(discovered.has('darwin-arm64'), 'finds the nested verdict');
assert.strictEqual(readVerdicts(pathModule.join(root, 'missing')).size, 0, 'a missing directory yields nothing');
fs.rmSync(root, { recursive: true, force: true });

console.log('verdict discovery tests passed');
