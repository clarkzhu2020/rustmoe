import process from 'node:process';

const query = process.argv.slice(2).join(' ').trim();
if (!query) { console.error('usage: node search.mjs "query"'); process.exit(2); }
const url = 'https://www.bing.com/search?q=' + encodeURIComponent(query);
const res = await fetch(url, { headers: { 'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36', 'accept-language': 'en-US,en;q=0.8' } });
if (!res.ok) { console.error('search HTTP ' + res.status); process.exit(1); }
const html = await res.text();
const results = [];
const blocks = html.split('<li class="b_algo"').slice(1);
for (const block of blocks) {
  const titleMatch = block.match(/<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/i);
  if (!titleMatch) continue;
  let link = titleMatch[1].replace(/&amp;/g, '&');
  const title = titleMatch[2].replace(/<[^>]+>/g, '').replace(/&amp;/g, '&').trim();
  const snippetMatch = block.match(/<p[^>]*class="b_lineclamp[^"]*"[^>]*>([\s\S]*?)<\/p>/i) || block.match(/<div class="b_caption"[^>]*>[\s\S]*?<p[^>]*>([\s\S]*?)<\/p>/i);
  const snippet = snippetMatch ? snippetMatch[1].replace(/<[^>]+>/g, '').replace(/&amp;/g, '&').trim() : '';
  if (link.startsWith('https://www.bing.com/ck/a')) {
    const u = new URL(link).searchParams.get('u');
    if (u && u.startsWith('a1')) {
      try { link = Buffer.from(u.slice(2), 'base64url').toString('utf8'); } catch {}
    }
  }
  results.push({ title, url: link, snippet });
  if (results.length >= 8) break;
}
console.log(JSON.stringify({ query, results }, null, 2));
