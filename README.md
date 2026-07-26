# TeluguPanchangamDaily — Static Site

Static website for **TeluguPanchangamDaily.com** — daily Telugu Panchangam, city-specific timings,
festival/muhurtham dates, and SEO content pages. Deployed via **Cloudflare Pages** (GitHub-connected);
routing and headers are handled by `_redirects` and `_headers`.

## Layout
- `index.html` — homepage / main panchangam, and the template city pages are generated from.
- `<city>/index.html` — ~80 generated city pages (e.g. `hyderabad/`, `london/`, `new-york/`).
- `telugu-panchangam-<month>-2026/`, `dasara-2026/`, `deepavali-2026/`, `muhurtham-dates-2026/`, … —
  generated month/festival/topic content pages.
- `cities/` — city index; `about.html`, `contact.html`, `privacy.html`, `disclaimer.html`,
  `advertise.html`, `404.html` — static pages.
- `city-content.json` — per-city content data consumed by the generator.
- `generate-city-pages.js`, `generate-2026-pages.js` — Node build scripts (no dependencies; they use
  only Node's built-in `fs`/`path` and self-locate via `__dirname`).
- Assets: `favicon*`, `apple-touch-icon.png`, `og-image.jpg`, `manifest.json`, `sw.js`.
- SEO / deploy: `sitemap.xml`, `sitemap-page.html`, `robots.txt`, `ads.txt`, `_headers`, `_redirects`,
  `bing-urls.txt`, and the IndexNow key file (`<guid>.txt`).
- Project docs: `GROWTH-STRATEGY.md`, `QA-PROMPT.md`, `CODEX-PROMPT.md`, `OUTREACH-TEMPLATES.md`.

## Build
Run the generators from this folder (they resolve all paths relative to their own location):
```bash
node generate-city-pages.js     # rebuild city pages; update sitemap.xml, sitemap-page.html, _redirects
node generate-2026-pages.js     # rebuild 2026 month/festival pages
```
No `npm install` is required.

## Serve locally
```bash
npx serve -p 3000 .
# or, with no network dependency:
python -m http.server 3000
```
Open <http://localhost:3000>. Serve **this folder as the web root** so absolute paths (`/...`) resolve.

## Deploy
Cloudflare Pages serves this folder's contents directly; `_redirects` and `_headers` are applied
automatically. Push to the connected GitHub branch to trigger a deploy.
