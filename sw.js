/* Telugu Panchangam Daily — Service Worker
   Strategy:
   - HTML/navigation: network-first (content must stay fresh), cache fallback for offline
   - CDN libraries & Google Fonts: stale-while-revalidate (fast repeat loads)
   - Ads/analytics: never intercepted
*/
const CACHE_VERSION = 'tpd-v1';
const RUNTIME_CACHE = CACHE_VERSION + '-runtime';

const CACHEABLE_CDN = [
  'cdn.jsdelivr.net',
  'unpkg.com',
  'fonts.googleapis.com',
  'fonts.gstatic.com'
];

const NEVER_CACHE = [
  'googlesyndication.com',
  'doubleclick.net',
  'google.com',
  'googletagservices.com',
  'googleadservices.com',
  'adtrafficquality.google',
  'cloudflareinsights.com',
  'fundingchoicesmessages.google.com'
];

self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => !k.startsWith(CACHE_VERSION)).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Never touch ads, analytics, or non-GET requests
  if (event.request.method !== 'GET') return;
  if (NEVER_CACHE.some(d => url.hostname.includes(d))) return;

  // HTML navigations: network-first with cache fallback (offline support)
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request)
        .then(resp => {
          const copy = resp.clone();
          caches.open(RUNTIME_CACHE).then(c => c.put(event.request, copy));
          return resp;
        })
        .catch(() => caches.match(event.request))
    );
    return;
  }

  // CDN libs and fonts: stale-while-revalidate
  if (CACHEABLE_CDN.some(d => url.hostname.includes(d)) || url.origin === self.location.origin) {
    event.respondWith(
      caches.match(event.request).then(cached => {
        const network = fetch(event.request).then(resp => {
          if (resp && resp.status === 200) {
            const copy = resp.clone();
            caches.open(RUNTIME_CACHE).then(c => c.put(event.request, copy));
          }
          return resp;
        }).catch(() => cached);
        return cached || network;
      })
    );
  }
});
