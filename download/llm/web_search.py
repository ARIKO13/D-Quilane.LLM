"""
D'QUILANE — Web Search Skills
==============================
Pure-Python web scraper. No API keys, no external dependencies.
Uses:
  - DuckDuckGo HTML endpoint (no JS, no API key needed)
  - Wikipedia REST API (for factual Q&A, multilingual)
  - HTTP via stdlib urllib only

Modes:
  - search(query, n=5)         → list of {title, url, snippet}
  - fetch_page(url)             → clean text from a URL
  - wikipedia(query, lang='id') → summary + url
  - cached_lookup(query)        → check cache first (avoid re-scraping)

Cache:
  - JSON file at /home/z/my-project/download/llm/web_cache.json
  - Keyed by normalized query
  - TTL: 24 hours (configurable)
"""
import urllib.request
import urllib.parse
import re
import json
import html
import time
import os
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================
CACHE_PATH = '/home/z/my-project/download/llm/web_cache.json'
CACHE_TTL_SECONDS = 24 * 3600  # 24 hours
USER_AGENT = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
TIMEOUT = 10  # seconds per request


# ============================================================
# HTTP UTILITY
# ============================================================
def _http_get(url, headers=None):
    """HTTP GET with custom UA, returns text or raises."""
    req = urllib.request.Request(url)
    req.add_header('User-Agent', USER_AGENT)
    req.add_header('Accept', 'text/html,application/json,*/*')
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        data = resp.read()
        # Try utf-8 first, fallback to latin-1
        try:
            return data.decode('utf-8')
        except UnicodeDecodeError:
            return data.decode('latin-1', errors='replace')


def _strip_html(text):
    """Crude HTML → text (no BeautifulSoup)."""
    # Remove scripts and styles
    text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', text,
                  flags=re.DOTALL | re.IGNORECASE)
    # Remove tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode HTML entities
    text = html.unescape(text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================
# CACHE
# ============================================================
def _load_cache():
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache):
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"[cache save error] {e}")


def _normalize_query(q):
    return re.sub(r'\s+', ' ', q.lower().strip())


def cached_lookup(query):
    """Return cached result dict if fresh (within TTL), else None."""
    cache = _load_cache()
    key = _normalize_query(query)
    entry = cache.get(key)
    if not entry:
        return None
    age = time.time() - entry.get('ts', 0)
    if age > CACHE_TTL_SECONDS:
        return None
    # entry = {ts, results} — return the inner results dict
    return entry.get('results')


def _store_cache(query, results):
    cache = _load_cache()
    key = _normalize_query(query)
    cache[key] = {
        'ts': time.time(),
        'results': results,
    }
    _save_cache(cache)


# ============================================================
# HACKER NEWS ALGOLIA API (always works, no auth)
# ============================================================
def search_hn(query, n=5):
    """
    Search Hacker News via Algolia API.
    Always works in any sandbox (no auth, no rate-limit).
    Best for tech / programming / news queries.
    """
    url = 'https://hn.algolia.com/api/v1/search?' + urllib.parse.urlencode({
        'query': query,
        'tags': 'story',
        'hitsPerPage': n,
    })
    try:
        text = _http_get(url, headers={'Accept': 'application/json'})
        data = json.loads(text)
        results = []
        for hit in data.get('hits', [])[:n]:
            results.append({
                'title': hit.get('title', ''),
                'url': hit.get('url') or f'https://news.ycombinator.com/item?id={hit.get("objectID")}',
                'snippet': f"Hacker News score: {hit.get('points', 0)} | comments: {hit.get('num_comments', 0)}",
                'source': 'hackernews',
            })
        return results
    except Exception as e:
        return [{'error': f'HN search failed: {e}'}]


# ============================================================
# BING SEARCH (best-effort, often bot-blocked)
# ============================================================
def search_bing(query, n=5):
    """
    Search Bing HTML. Best-effort — Bing often serves bot-version HTML
    without parseable results. Returns [] if no results found.
    """
    base = 'https://www.bing.com/search'
    params = {'q': query, 'count': n}
    url = base + '?' + urllib.parse.urlencode(params)

    try:
        text = _http_get(url)
    except Exception as e:
        return []  # silent fail — Bing is best-effort only

    results = []

    # Try 1: b_algo blocks (full Bing HTML — rare in sandbox)
    blocks = re.findall(r'<li class="b_algo"[^>]*>(.*?)</li>', text, re.DOTALL)
    for block in blocks[:n]:
        m_link = re.search(
            r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            block, re.DOTALL
        )
        if not m_link:
            continue
        url_ = m_link.group(1)
        title = _strip_html(m_link.group(2))
        m_snip = re.search(r'<p[^>]*>(.*?)</p>', block, re.DOTALL)
        snippet = _strip_html(m_snip.group(1)) if m_snip else ''
        if url_.startswith('/search?') or 'bing.com' in url_:
            continue
        results.append({'title': title, 'url': url_, 'snippet': snippet,
                         'source': 'bing'})

    return results


# ============================================================
# DUCKDUCKGO SEARCH (fallback, often blocked)
# ============================================================
def search_ddg(query, n=5):
    """Search DuckDuckGo HTML. Often blocks sandbox IPs."""
    base = 'https://html.duckduckgo.com/html/'
    params = {'q': query, 'kl': 'id-id'}
    url = base + '?' + urllib.parse.urlencode(params)

    try:
        text = _http_get(url)
    except Exception as e:
        return []  # silent fail

    results = []
    blocks = re.findall(
        r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        text, re.DOTALL
    )
    for url_, title_html, snippet_html in blocks[:n]:
        title = _strip_html(title_html)
        snippet = _strip_html(snippet_html)
        if 'uddg=' in url_:
            m = re.search(r'uddg=([^&]+)', url_)
            if m:
                url_ = urllib.parse.unquote(m.group(1))
        elif url_.startswith('//'):
            url_ = 'https:' + url_
        results.append({'title': title, 'url': url_, 'snippet': snippet,
                         'source': 'duckduckgo'})
    return results


# ============================================================
# WIKIPEDIA
# ============================================================
def wikipedia(query, lang='id'):
    """
    Search Wikipedia REST API. Returns summary + page URL.
    Tries multiple languages: id, en, etc.
    """
    base = f'https://{lang}.wikipedia.org/api/rest_v1/page/summary/'
    # First, search for the page title
    search_url = f'https://{lang}.wikipedia.org/w/api.php'
    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': query,
        'format': 'json',
        'srlimit': 1,
    }
    search_url_full = search_url + '?' + urllib.parse.urlencode(params)

    try:
        text = _http_get(search_url_full,
                         headers={'Accept': 'application/json'})
        data = json.loads(text)
        if not data.get('query', {}).get('search'):
            # Try English fallback
            if lang != 'en':
                return wikipedia(query, lang='en')
            return None
        title = data['query']['search'][0]['title']
    except Exception as e:
        return {'error': f'Wikipedia search failed: {e}'}

    # Fetch summary
    summary_url = base + urllib.parse.quote(title)
    try:
        text = _http_get(summary_url,
                         headers={'Accept': 'application/json'})
        data = json.loads(text)
        return {
            'title': data.get('title', title),
            'extract': data.get('extract', ''),
            'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
            'thumbnail': data.get('thumbnail', {}).get('source', ''),
        }
    except Exception as e:
        return {'error': f'Wikipedia fetch failed: {e}'}


# ============================================================
# MAIN SEARCH (with cache)
# ============================================================
def search(query, n=5, use_wiki=True, force_refresh=False):
    """
    Main search function. Returns dict with:
      {
        'query': str,
        'cached': bool,
        'web_results': [{title, url, snippet}, ...],
        'wiki': {title, extract, url, thumbnail} or None,
      }
    """
    # Check cache
    if not force_refresh:
        cached = cached_lookup(query)
        if cached:
            cached['cached'] = True
            return cached

    results = {
        'query': query,
        'cached': False,
        'ts': time.time(),
        'web_results': [],
        'wiki': None,
        'sources_tried': [],
    }

    # Source 1: Wikipedia (factual, multilingual — best for knowledge Q&A)
    if use_wiki:
        results['sources_tried'].append('wikipedia')
        wiki = wikipedia(query, lang='id')
        if wiki and 'extract' in wiki:
            results['wiki'] = wiki

    # Source 2: Hacker News Algolia (always works — for tech/news queries)
    results['sources_tried'].append('hackernews')
    hn_results = search_hn(query, n=n)
    if hn_results and not any('error' in r for r in hn_results):
        results['web_results'].extend(hn_results)

    # Source 3: Bing (best-effort, may serve bot version)
    results['sources_tried'].append('bing')
    bing_results = search_bing(query, n=n)
    if bing_results:
        results['web_results'].extend(bing_results)

    # Source 4: DuckDuckGo (fallback — often times out)
    if len(results['web_results']) < n:
        results['sources_tried'].append('duckduckgo')
        ddg_results = search_ddg(query, n=n)
        if ddg_results:
            results['web_results'].extend(ddg_results)

    # Deduplicate by URL
    seen = set()
    unique = []
    for r in results['web_results']:
        if 'error' in r:
            continue
        u = r.get('url', '')
        if u and u not in seen:
            seen.add(u)
            unique.append(r)
    results['web_results'] = unique[:n]

    # Cache
    _store_cache(query, results)
    return results


# ============================================================
# FETCH A WEB PAGE
# ============================================================
def fetch_page(url, max_chars=2000):
    """Fetch a URL and extract clean text (truncated to max_chars)."""
    try:
        text = _http_get(url)
        clean = _strip_html(text)
        # Take first paragraph-ish content
        # Skip common boilerplate by finding first substantial sentence
        if len(clean) > max_chars:
            # Try to cut at a sentence boundary
            cut = clean[:max_chars]
            last_period = cut.rfind('. ')
            if last_period > max_chars // 2:
                cut = cut[:last_period + 1]
            clean = cut + '...'
        return {
            'url': url,
            'text': clean,
            'chars': len(clean),
        }
    except Exception as e:
        return {'error': f'Fetch failed: {e}'}


# ============================================================
# TRIGGER DETECTION
# ============================================================
TRIGGER_KEYWORDS = [
    # Indonesian
    'terbaru', 'terkini', 'sekarang', 'hari ini', 'update',
    'cari', 'search', 'googling', 'tanya web',
    'internet', 'online', 'web', 'berita',
    # English
    'latest', 'recent', 'today', 'now', 'current',
    'news', 'online',
]

EXPLICIT_TRIGGERS = [
    'search:', 'cari:', 'google:', 'web:',
    'search ', 'cari ', 'google ',
]


def needs_web_search(prompt):
    """Detect if prompt should trigger web search."""
    p = prompt.lower().strip()
    # Explicit trigger
    for t in EXPLICIT_TRIGGERS:
        if p.startswith(t):
            return True, p[len(t):].strip()
    # Keyword trigger
    for kw in TRIGGER_KEYWORDS:
        if kw in p:
            return True, p
    return False, prompt


def format_search_response(results):
    """Format search results as a readable string for inference output."""
    if not results:
        return "Tidak ada hasil."

    lines = []
    # Header
    src = ', '.join(results.get('sources_tried', []))
    lines.append(f"[Web Search] sources: {src} | "
                 f"{'CACHED' if results.get('cached') else 'FRESH'}")
    lines.append('')

    if results.get('wiki'):
        wiki = results['wiki']
        lines.append(f"[Wikipedia] {wiki.get('title', '')}")
        if wiki.get('extract'):
            extract = wiki['extract']
            if len(extract) > 500:
                extract = extract[:500] + '...'
            lines.append(extract)
        if wiki.get('url'):
            lines.append(f"Sumber: {wiki['url']}")
        lines.append('')

    web = results.get('web_results', [])
    if web:
        lines.append(f"[Web Results] {len(web)} hasil:")
        for i, r in enumerate(web[:5], 1):
            if r.get('error'):
                lines.append(f"  {i}. Error: {r['error']}")
            else:
                src_tag = f" [{r.get('source', '')}]" if r.get('source') else ''
                lines.append(f"  {i}.{src_tag} {r.get('title', '')[:100]}")
                if r.get('snippet'):
                    snippet = r['snippet'][:200]
                    lines.append(f"     {snippet}")
                if r.get('url'):
                    lines.append(f"     {r['url']}")
            lines.append('')

    if not web and not results.get('wiki'):
        lines.append('Tidak ada hasil dari sumber manapun.')
        lines.append('Sandbox mungkin memblokir DuckDuckGo/Bing.')
        lines.append('Coba tambahkan prefix "search:" di depan prompt untuk paksa search.')

    return '\n'.join(lines)


# ============================================================
# CLI
# ============================================================
def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python web_search.py <query> [--wiki-only] [--no-cache]")
        print("       python web_search.py \"apa kabar hari ini\"")
        print("       python web_search.py \"ibukota Indonesia\" --wiki-only")
        sys.exit(1)

    query = sys.argv[1]
    wiki_only = '--wiki-only' in sys.argv
    no_cache = '--no-cache' in sys.argv

    print(f"[Search] {query}")
    t0 = time.time()
    results = search(query, n=5, use_wiki=not wiki_only or True,
                     force_refresh=no_cache)
    elapsed = time.time() - t0

    print(f"  Completed in {elapsed:.2f}s  (cached: {results.get('cached', False)})")
    print()
    print(format_search_response(results))


if __name__ == '__main__':
    main()
