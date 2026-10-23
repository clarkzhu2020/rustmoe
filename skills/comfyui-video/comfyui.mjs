import process from 'node:process';
import fs from 'node:fs';

const base = (process.env.COMFYUI_URL || 'http://127.0.0.1:8188').replace(/\/$/, '');
const action = process.argv[2];
const arg = process.argv[3];
if (!action) { console.error('usage: comfyui.mjs status|queue|history [arg]'); process.exit(2); }
async function get(path) {
  const r = await fetch(base + path);
  if (!r.ok) throw new Error('HTTP ' + r.status + ' ' + path);
  return r.json();
}
try {
  if (action === 'status') {
    console.log(JSON.stringify({ system: await get('/system_stats'), queue: await get('/queue') }, null, 2));
  } else if (action === 'queue') {
    if (!arg) throw new Error('queue requires workflow.json');
    const workflow = JSON.parse(fs.readFileSync(arg, 'utf8'));
    const r = await fetch(base + '/prompt', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ prompt: workflow }) });
    if (!r.ok) throw new Error('HTTP ' + r.status + ' ' + (await r.text()));
    console.log(JSON.stringify(await r.json(), null, 2));
  } else if (action === 'history') {
    if (!arg) throw new Error('history requires prompt_id');
    console.log(JSON.stringify(await get('/history/' + encodeURIComponent(arg)), null, 2));
  } else {
    throw new Error('unknown action ' + action);
  }
} catch (e) { console.error(e.message); process.exit(1); }
