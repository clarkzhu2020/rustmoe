import process from 'node:process';

const url = process.argv[2];
if (!url) { console.error('usage: node fetch.mjs URL'); process.exit(2); }
const res = await fetch(url, { headers: { 'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36' } });
if (!res.ok) { console.error('HTTP ' + res.status); process.exit(1); }
const buf = Buffer.from(await res.arrayBuffer());
const text = buf.toString('utf8');
process.stdout.write(text.slice(0, 200000));
