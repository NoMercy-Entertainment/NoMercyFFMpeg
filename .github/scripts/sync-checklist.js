#!/usr/bin/env node
// Ticks the PR test-checklist from verification verdicts, and posts the
// evidence behind every tick.
//
// The rule this file exists to enforce: a box is ticked only when a verdict
// file proves that THIS commit's artifact was tested on hardware that could
// actually run it. Job success is not evidence — a job can pass for reasons
// unrelated to the binary. Anything unproven is left unticked, which keeps the
// merge blocked exactly as an untested platform should.
'use strict';

const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const { START, END } = require('./check-checklist.js');

const COMMENT_MARKER = '<!-- RC-VERIFICATION-REPORT -->';

// Maps a checklist line to the verdict that is allowed to tick it. The hardware
// line is special: it is satisfied by acceleration actually exercised somewhere
// in the fleet, not by one platform.
const PLATFORM_LINE = /^(linux-x86_64|linux-aarch64|windows-x86_64|windows-aarch64|darwin-x86_64|darwin-arm64|freebsd-x86_64)\b/;
const HARDWARE_LINE = /hardware acceleration/i;
const ACCELERATORS = ['nvenc', 'amf', 'vpl', 'videotoolbox'];

// Walks nested directories on purpose. download-artifact flattens with
// merge-multiple, but `gh run download` gives every artifact its own folder —
// and a non-recursive read of that layout finds nothing, silently leaves every
// box unticked, and reports six healthy platforms as unverified.
function findVerdictFiles(directory) {
  const found = [];
  if (!fs.existsSync(directory)) return found;

  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const full = path.join(directory, entry.name);
    if (entry.isDirectory()) found.push(...findVerdictFiles(full));
    else if (entry.name.endsWith('.json')) found.push(full);
  }
  return found;
}

function readVerdicts(directory) {
  const verdicts = new Map();

  for (const file of findVerdictFiles(directory)) {
    let parsed;
    try {
      parsed = JSON.parse(fs.readFileSync(file, 'utf8'));
    } catch (error) {
      console.error(`⚠️  ${path.basename(file)}: unreadable (${error.message}) — treated as missing`);
      continue;
    }
    if (parsed && parsed.platform) verdicts.set(parsed.platform, parsed);
  }
  return verdicts;
}

// Every condition here is a reason a tick would be a lie.
function assess(verdict, expected) {
  if (!verdict) return { ok: false, why: 'no verdict produced (runner offline or job never ran)' };
  if (verdict.verdict !== 'pass') {
    const failed = (verdict.report?.tests || []).filter((t) => t.status === 'failed');
    const names = failed.map((t) => t.name).join(', ');
    const stage = verdict.failed_stage ? `${verdict.failed_stage}: ${verdict.detail}` : '';
    return { ok: false, why: names ? `failed: ${names}` : stage || 'suite reported failure' };
  }
  // Absent is not the same as matching. These two fields are the entire reason
  // a verdict from an older push cannot tick a box for a newer one, so a
  // verdict that does not carry them has not proved anything about which build
  // it tested.
  if (!verdict.commit) {
    return { ok: false, why: 'verdict does not say which commit it tested' };
  }
  if (expected.headSha && verdict.commit !== expected.headSha) {
    return { ok: false, why: `verdict is for commit ${verdict.commit.slice(0, 8)}, PR head is ${expected.headSha.slice(0, 8)}` };
  }
  if (!verdict.tag) {
    return { ok: false, why: 'verdict does not say which release it tested' };
  }
  if (expected.tag && verdict.tag !== expected.tag) {
    return { ok: false, why: `verdict is for ${verdict.tag}, expected ${expected.tag}` };
  }
  if (!verdict.integrity || verdict.integrity.verified !== true) {
    return { ok: false, why: 'artifact digest was never checked against the manifest' };
  }
  if (!verdict.report || (verdict.report.totals?.total || 0) === 0) {
    return { ok: false, why: 'verdict carries no test results' };
  }
  if ((verdict.report.totals?.failed || 0) > 0) {
    return { ok: false, why: `${verdict.report.totals.failed} test(s) failed` };
  }
  return { ok: true, why: '' };
}

// The hardware line covers NVENC / AMF / VPL. Nothing in the fleet can
// positively exercise AMF or VPL, so it is satisfied by at least one
// accelerator genuinely encoding, with the untestable ones reported as such.
// Takes the ASSESSED results, not the raw verdict map, and does the filtering
// itself. Passing accepted verdicts in from the caller worked, but it left the
// linkage as something to remember: reading the raw map here would once again
// let a verdict refused for a platform box tick the hardware box, which is the
// same lie by a different route. Demanding the assessed shape makes the wrong
// call fail rather than pass quietly.
function assessHardware(results) {
  const exercised = [];
  const unavailable = [];

  const accepted = [...results.values()]
    .filter((entry) => entry && entry.assessment && entry.assessment.ok && entry.verdict)
    .map((entry) => entry.verdict);

  for (const verdict of accepted) {
    for (const test of verdict.report?.tests || []) {
      const name = test.name.toLowerCase();
      if (!ACCELERATORS.includes(name)) continue;
      if (test.status === 'passed') exercised.push(`${name} on ${verdict.platform}`);
      if (test.status === 'failed') {
        return { ok: false, why: `${test.name} failed on ${verdict.platform}`, exercised, unavailable };
      }
      if (test.status === 'skipped') unavailable.push(`${name}: ${test.reason}`);
    }
  }

  if (exercised.length === 0) {
    return { ok: false, why: 'no accelerator was exercised anywhere in the fleet', exercised, unavailable };
  }
  return { ok: true, why: '', exercised, unavailable };
}

function rewriteChecklist(body, decide) {
  const start = body.indexOf(START);
  const end = body.indexOf(END);
  if (start === -1 || end === -1 || end < start) return null;

  const head = body.slice(0, start + START.length);
  const section = body.slice(start + START.length, end);
  const tail = body.slice(end);

  const rewritten = section
    .split(/\r?\n/)
    .map((line) => {
      const match = line.match(/^(\s*-\s*\[)( |x|X)(\]\s*)(.*)$/);
      if (!match) return line;
      const label = match[4];
      // The current state has to be handed to decide(). label is the text AFTER
      // the checkbox, so anything testing it for "[x]" can never match, and the
      // unrecognised-line branch would clear a box a human ticked.
      const wasChecked = match[2].toLowerCase() === 'x';
      return `${match[1]}${decide(label, wasChecked) ? 'x' : ' '}${match[3]}${label}`;
    })
    .join('\n');

  return head + rewritten + tail;
}

function buildComment(results, hardware, context) {
  const rows = [...results.entries()].map(([platform, { assessment, verdict }]) => {
    const icon = assessment.ok ? '✅' : '❌';
    const totals = verdict?.report?.totals;
    const counts = totals ? `${totals.passed} passed, ${totals.skipped} skipped, ${totals.failed} failed` : '—';
    const host = verdict?.report?.host
      ? `${verdict.report.host.os}/${verdict.report.host.arch}`
      : '—';
    const native = verdict?.execution
      ? (verdict.execution.native ? 'native' : `via ${verdict.execution.translation}`)
      : '—';
    const digest = verdict?.integrity?.sha256 ? `\`${verdict.integrity.sha256.slice(0, 12)}\`` : '—';
    const note = assessment.ok ? '' : ` — ${assessment.why}`;
    return `| ${icon} ${platform} | ${counts} | ${host} | ${native} | ${digest}${note} |`;
  });

  const lines = [
    COMMENT_MARKER,
    `## RC verification — ${context.tag}`,
    '',
    `Ran the full suite on real hardware for commit \`${context.headSha.slice(0, 8)}\`.`,
    'Each row is the machine that executed the binary, not a summary of a build job.',
    '',
    '| Platform | Result | Host | Execution | Artifact SHA-256 |',
    '|---|---|---|---|---|',
    ...rows,
    '',
    `**Hardware acceleration:** ${hardware.ok ? '✅' : '❌'} ${hardware.ok ? 'verified' : hardware.why}`,
  ];

  if (hardware.exercised.length > 0) {
    lines.push('', 'Exercised on real hardware:', ...hardware.exercised.map((item) => `- ${item}`));
  }
  if (hardware.unavailable.length > 0) {
    const unique = [...new Set(hardware.unavailable)];
    lines.push(
      '',
      'Not exercised — no such hardware in the verification fleet:',
      ...unique.map((item) => `- ${item}`),
    );
  }

  lines.push('', `[Full logs and verdict artifacts](${context.runUrl})`);
  return lines.join('\n');
}

function gh(args, input) {
  return execFileSync('gh', args, {
    encoding: 'utf8',
    input,
    maxBuffer: 32 * 1024 * 1024,
  });
}

function main() {
  const directory = process.argv[2] || 'verdicts';
  const context = {
    repo: process.env.REPO,
    pr: process.env.PR,
    tag: process.env.TAG || '',
    headSha: process.env.HEAD_SHA || '',
    runUrl: process.env.RUN_URL || '',
  };

  if (!context.repo || !context.pr) {
    console.error('❌ REPO and PR must be set.');
    process.exit(1);
  }

  const verdicts = readVerdicts(directory);
  const results = new Map();
  for (const platform of [
    'linux-x86_64', 'linux-aarch64', 'windows-x86_64', 'windows-aarch64',
    'darwin-x86_64', 'darwin-arm64', 'freebsd-x86_64',
  ]) {
    const verdict = verdicts.get(platform);
    results.set(platform, { verdict, assessment: assess(verdict, context) });
  }
  const hardware = assessHardware(results);

  const body = JSON.parse(gh(['pr', 'view', context.pr, '--repo', context.repo, '--json', 'body'])).body || '';

  const updated = rewriteChecklist(body, (label, wasChecked) => {
    const platformMatch = label.match(PLATFORM_LINE);
    if (platformMatch) return results.get(platformMatch[1])?.assessment.ok === true;
    if (HARDWARE_LINE.test(label)) return hardware.ok;
    // An unrecognised checklist line is never auto-ticked, and never cleared
    // either. A human added it for a reason this script cannot verify, and
    // clearing it would block the merge with no way for them to tick out of it.
    console.error(`ℹ️  leaving unrecognised checklist item untouched: ${label}`);
    return wasChecked;
  });

  if (updated === null) {
    console.error('❌ Test-checklist delimiters not found in the PR body.');
    process.exit(1);
  }

  const comment = buildComment(results, hardware, context);

  // Rehearsal mode: prove the whole pipeline — runners, artifacts, verdicts and
  // the tick decision — without editing someone's open release PR.
  if (process.env.DRY_RUN === 'true') {
    console.log('── DRY RUN — nothing will be written to the PR ──\n');
    console.log(comment);
    console.log('\n── checklist that would be written ──');
    for (const [platform, { assessment }] of results) {
      console.log(`  [${assessment.ok ? 'x' : ' '}] ${platform}${assessment.ok ? '' : ` — ${assessment.why}`}`);
    }
    console.log(`  [${hardware.ok ? 'x' : ' '}] hardware acceleration`);
    console.log(updated === body ? '\nPR body unchanged.' : '\nPR body would change.');
    return;
  }

  if (updated !== body) {
    gh(['pr', 'edit', context.pr, '--repo', context.repo, '--body-file', '-'], updated);
    console.log('✅ PR checklist updated.');
  } else {
    console.log('ℹ️  PR checklist already matches the verdicts.');
  }

  // One sticky comment, edited in place, so a PR that is pushed to repeatedly
  // does not accumulate a wall of near-identical reports.
  const existing = JSON.parse(
    gh(['pr', 'view', context.pr, '--repo', context.repo, '--json', 'comments']),
  ).comments || [];
  const previous = existing.filter((c) => (c.body || '').includes(COMMENT_MARKER)).pop();

  if (previous && previous.url) {
    const id = previous.url.split('-').pop();
    gh(['api', '-X', 'PATCH', `repos/${context.repo}/issues/comments/${id}`,
      '-F', 'body=@-'], comment);
    console.log('✅ Evidence comment updated.');
  } else {
    gh(['pr', 'comment', context.pr, '--repo', context.repo, '--body-file', '-'], comment);
    console.log('✅ Evidence comment posted.');
  }

  const failed = [...results.values()].filter((r) => !r.assessment.ok);
  if (failed.length > 0 || !hardware.ok) {
    console.error(`❌ ${failed.length} platform(s) unverified; checklist left incomplete.`);
    process.exit(1);
  }
  console.log('✅ Every checklist item verified.');
}

if (require.main === module) main();

module.exports = { assess, assessHardware, rewriteChecklist, buildComment, readVerdicts, PLATFORM_LINE };
