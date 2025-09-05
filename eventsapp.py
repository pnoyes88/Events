# app.py
# Weekend Events Finder (patched)
# - Adds seed URLs (Eventbrite/Meetup/AllEvents/Patch/city homepage guess)
# - Makes search step resilient and visible
# - Relaxes weekend filtering & improves parsing tolerance
# - Includes a dependency guard for BeautifulSoup

import re
import io
import csv
import json
import unicodedata
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse

import streamlit as st

# Dependency guard for BeautifulSoup
try:
    from bs4 import BeautifulSoup
except ImportError:
    st.error("Missing dependency: 'beautifulsoup4'. Install with `pip install beautifulsoup4` "
             "or add it to requirements.txt.")
    st.stop()

import requests
from dateutil import parser as dateparser
from dateutil import tz

# DuckDuckGo search (graceful if missing)
try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None

# ICS parsing (optional)
try:
    import icalendar
except Exception:
    icalendar = None


# ----------------------
# Config
# ----------------------
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; WeekendEventsBot/1.0; +https://streamlit.io/)",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


# ----------------------
# Weekend window
# ----------------------
def upcoming_weekend_range(user_tz="America/New_York"):
    """Return (start_dt, end_dt) for the upcoming weekend in timezone user_tz: Fri 00:00 → Sun 23:59:59."""
    tzinfo = tz.gettz(user_tz)
    now = datetime.now(tzinfo)
    days_until_friday = (4 - now.weekday()) % 7  # Mon=0 ... Sun=6; Friday=4
    friday = (now + timedelta(days=days_until_friday)).replace(hour=0, minute=0, second=0, microsecond=0)
    sunday_end = (friday + timedelta(days=2)).replace(hour=23, minute=59, second=59, microsecond=0)
    return friday, sunday_end


# ----------------------
# Search query generator
# ----------------------
def build_queries(town: str, state: str, extra_keywords: str = ""):
    base = f"{town}, {state}".strip()
    ek = f" {extra_keywords.strip()}" if extra_keywords.strip() else ""
    queries = [
        f"{base} events this weekend{ek}",
        f"{base} calendar events{ek}",
        f"{base} things to do this weekend{ek}",
        f"{base} library events{ek}",
        f"{base} parks and recreation events{ek}",
        f"{base} chamber of commerce events{ek}",
        f"{base} town hall events{ek}",
        f"site:.gov {base} events OR calendar{ek}",
        f"site:.org {base} events OR calendar{ek}",
        f"site:eventbrite.com {base} events{ek}",
        f"site:meetup.com {base} events{ek}",
        f"site:facebook.com/events {base}{ek}",
        f"site:allevents.in {base}{ek}",
        f"site:patch.com {base} events{ek}",
    ]
    seen, out = set(), []
    for q in queries:
        if q not in seen:
            out.append(q); seen.add(q)
    return out


# ----------------------
# Seed aggregators (work even if search returns nothing)
# ----------------------
def _slugify(s: str):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

def aggregator_seed_urls(city: str, state: str):
    city_slug = _slugify(city)
    state_abbr = state.upper()[:2]  # assume 2-letter; still works for full names in many cases
    state_slug = _slugify(state)

    # Eventbrite "this weekend"
    eb = f"https://www.eventbrite.com/d/{state_abbr.lower()}--{city_slug}/events--this-weekend/"

    # Meetup discovery (HTML has enough SSR links to crawl further)
    mu = f"https://www.meetup.com/find/?source=EVENTS&location=us--{state_abbr.lower()}--{city_slug}&startDateRange=this-weekend"

    # AllEvents "this weekend"
    ae = f"https://allevents.in/{city_slug}/this-weekend"

    # Patch calendar (not every town, but cheap to try)
    patch = f"https://patch.com/{state_slug}/{city_slug}/calendar"

    # Many municipalities: https://city.state.us (worth probing for calendar links)
    city_home_guess = f"https://www.{city_slug}.{state_abbr.lower()}.us/"

    return [eb, mu, ae, patch, city_home_guess]


# ----------------------
# Fetching helpers
# ----------------------
def fetch_url(url: str, timeout=12):
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        if r.status_code == 200 and r.content:
            return r.text, r.url
    except Exception:
        return None, url
    return None, url


# ----------------------
# JSON-LD Event extraction
# ----------------------
def _flatten_jsonld(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _flatten_jsonld(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _flatten_jsonld(item)

def _looks_like_event_type(jtype):
    if isinstance(jtype, str):
        return jtype.lower() == "event"
    if isinstance(jtype, list):
        return any(isinstance(t, str) and t.lower() == "event" for t in jtype)
    return False

def parse_jsonld_events(html: str, base_url: str):
    soup = BeautifulSoup(html, "html.parser")
    events = []
    for tag in soup.find_all("script", type=lambda t: t and "ld+json" in t.lower()):
        raw = tag.string or tag.get_text() or ""
        if not raw.strip():
            continue
        candidates = []
        try:
            candidates.append(json.loads(raw))
        except Exception:
            parts = re.split(r"}\s*{", raw.strip())
            if len(parts) > 1:
                rebuilt = []
                for i, p in enumerate(parts):
                    if i == 0: rebuilt.append(p + "}")
                    elif i == len(parts)-1: rebuilt.append("{" + p)
                    else: rebuilt.append("{" + p + "}")
                for chunk in rebuilt:
                    try:
                        candidates.append(json.loads(chunk))
                    except Exception:
                        pass
        for data in candidates:
            for node in _flatten_jsonld(data):
                if "@type" in node and _looks_like_event_type(node["@type"]):
                    ev = {
                        "name": node.get("name"),
                        "description": node.get("description"),
                        "url": node.get("url") or base_url,
                        "start": node.get("startDate") or node.get("start_time"),
                        "end": node.get("endDate") or node.get("end_time"),
                        "location": "",
                    }
                    loc = node.get("location")
                    if isinstance(loc, dict):
                        ev["location"] = loc.get("name") or loc.get("address") or ""
                    else:
                        ev["location"] = loc or ""
                    events.append(ev)
    return events


# ----------------------
# ICS discovery & parsing
# ----------------------
def discover_ics_links(html: str, base_url: str):
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if any(ext in href.lower() for ext in [".ics", "ical", "calendar.ics"]):
            links.append(urljoin(base_url, href))
    return list(dict.fromkeys(links))

def parse_ics_events(ics_text: str, source_url: str):
    if not icalendar:
        return []
    events = []
    try:
        cal = icalendar.Calendar.from_ical(ics_text)
        for comp in cal.walk():
            if comp.name == "VEVENT":
                ev = {
                    "name": str(comp.get("summary", "")),
                    "description": str(comp.get("description", "")),
                    "url": str(comp.get("url", "")) or source_url,
                    "location": str(comp.get("location", "")),
                    "start": None,
                    "end": None,
                }
                start = comp.get("dtstart")
                end = comp.get("dtend")
                if start:
                    try: ev["start"] = start.dt.isoformat()
                    except Exception: ev["start"] = str(start.dt)
                if end:
                    try: ev["end"] = end.dt.isoformat()
                    except Exception: ev["end"] = str(end.dt)
                events.append(ev)
    except Exception:
        pass
    return events


# ----------------------
# Heuristic fallback
# ----------------------
def heuristic_extract_events(html: str, base_url: str, town: str):
    soup = BeautifulSoup(html, "html.parser")
    candidates = []
    for sel in ["div", "li", "article", "section"]:
        for node in soup.find_all(sel):
            cls = " ".join(node.get("class", [])).lower()
            nid = (node.get("id") or "").lower()
            if any(k in cls or k in nid for k in ["event", "calendar", "whatson", "whats-on"]):
                text = " ".join(node.stripped_strings)
                if len(text) > 40:
                    candidates.append((node, text))
    date_patterns = [
        r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b.*?\b(?:\d{1,2}:\d{2}\s?(?:AM|PM))?",
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b.*?\b\d{1,2}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
    ]
    date_re = re.compile("|".join(date_patterns), re.IGNORECASE)
    events = []
    for node, text in candidates:
        link = node.find("a", href=True)
        url = urljoin(base_url, link["href"]) if link else base_url
        lines = [s for s in node.stripped_strings]
        if not lines:
            continue
        title = lines[0][:140]
        m = date_re.search(text)
        approx_date = m.group(0) if m else ""
        events.append({
            "name": title,
            "description": approx_date,
            "url": url,
            "location": town,
            "start": "",
            "end": "",
        })
    return events


# ----------------------
# Normalize & filter
# ----------------------
def to_dt(v, tzinfo):
    if not v:
        return None
    try:
        dt = dateparser.parse(v)
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=tzinfo)
        return dt.astimezone(tzinfo)
    except Exception:
        return None

def normalize_events(evts, tzname, source):
    tzinfo = tz.gettz(tzname)
    out = []
    for e in evts:
        start = to_dt(e.get("start"), tzinfo)
        end = to_dt(e.get("end"), tzinfo)
        out.append({
            "name": e.get("name") or "(untitled event)",
            "start": start,
            "end": end,
            "location": e.get("location") or "",
            "url": e.get("url") or source,
            "description": (e.get("description") or "").strip(),
            "source": source,
        })
    return out

def filter_to_weekend(events, wk_start, wk_end):
    """Relaxed filter: accept parsed datetimes inside Fri–Sun OR items that say Fri/Sat/Sun explicitly."""
    out = []
    for e in events:
        s = e["start"]
        if not s and e["description"]:
            # Another parse attempt for short forms like "Sat 9/7 2pm"
            s = to_dt(e["description"], wk_start.tzinfo)
            if s:
                e["start"] = s
        if not s:
            if re.search(r"\b(Fri(day)?|Sat(urday)?|Sun(day)?)\b", e["description"], re.I):
                out.append(e)
                continue
        if s and wk_start <= s <= wk_end:
            out.append(e)
    return out


# ----------------------
# Search & scrape pipeline
# ----------------------
def ddg_search_urls(query, max_results=6):
    urls = []
    if DDGS is None:
        return urls
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results, region="us-en", safesearch="moderate"):
                href = r.get("href") or r.get("link") or r.get("url")
                if href and href.startswith("http"):
                    urls.append(href)
    except Exception as e:
        # Surface the issue in the UI so you know search failed (not scraping)
        st.info(f"Search fallback engaged (DuckDuckGo error: {e})")
        return []
    # Dedup domain+path
    seen, res = set(), []
    for u in urls:
        key = urlparse(u)._replace(query="", fragment="").geturl()
        if key not in seen:
            res.append(u); seen.add(key)
    return res


def harvest_from_url(url, town, tzname, wk_start, wk_end, enable_ics=True):
    html, final_url = fetch_url(url)
    if not html:
        return []

    # JSON-LD
    jl = parse_jsonld_events(html, final_url)
    normalized = normalize_events(jl, tzname, final_url)
    weekend_hits = filter_to_weekend(normalized, wk_start, wk_end)

    # ICS
    if enable_ics and icalendar:
        ics_links = discover_ics_links(html, final_url)
        for ics_url in ics_links[:3]:
            try:
                txt, _ = fetch_url(ics_url)
                if txt:
                    ics_events = parse_ics_events(txt, ics_url)
                    normalized_ics = normalize_events(ics_events, tzname, ics_url)
                    weekend_hits.extend(filter_to_weekend(normalized_ics, wk_start, wk_end))
            except Exception:
                pass

    # Heuristic fallback
    if not weekend_hits:
        crude = heuristic_extract_events(html, final_url, town)
        normalized_crude = normalize_events(crude, tzname, final_url)
        weekend_words = re.compile(r"\b(Fri|Friday|Sat|Saturday|Sun|Sunday|weekend)\b", re.I)
        weekendish = [e for e in normalized_crude if e["start"] or weekend_words.search(e["description"])]
        weekend_hits.extend(weekendish)

    return weekend_hits


# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="Weekend Events Finder", page_icon="📅", layout="wide")
st.title("📅 Weekend Events Finder")
st.caption("Enter a town & state. I’ll search common local sources and extract events for the upcoming weekend.")

with st.sidebar:
    st.markdown("### Search Options")
    user_tz = st.text_input("Time zone (IANA)", value="America/New_York", help="Used to compute the weekend range.")
    max_results_per_query = st.slider("Max results per query", 3, 15, 10)
    enable_ics = st.checkbox("Parse .ics calendars when found", value=True)
    allow_facebook = st.checkbox("Include Facebook event links (often non-scrapable)", value=True)
    st.markdown("---")
    st.markdown("**Tip:** Add extra keywords like a venue or neighborhood.")

col1, col2 = st.columns([2, 1])
with col1:
    town = st.text_input("Town / City", placeholder="e.g., Boston")
    state = st.text_input("State (2-letter or full name)", placeholder="e.g., MA")
with col2:
    extra = st.text_input("Extra keywords (optional)", placeholder="e.g., farmers market, concerts")
    search_btn = st.button("🔎 Find Weekend Events", type="primary")

wk_start, wk_end = upcoming_weekend_range(user_tz)
st.markdown(
    f"**Weekend window:** {wk_start.strftime('%a %b %d, %Y %H:%M %Z')} → {wk_end.strftime('%a %b %d, %Y %H:%M %Z')}"
)

if search_btn:
    if not town or not state:
        st.error("Please enter both a town and a state.")
        st.stop()

    # ALWAYS include aggregator seeds
    seeds = aggregator_seed_urls(town, state)
    all_urls = list(seeds)

    # Gather URLs from search queries
    queries = build_queries(town, state, extra)
    progress = st.progress(0.0)
    status = st.empty()

    for i, q in enumerate(queries, start=1):
        status.write(f"Searching: _{q}_")
        urls = ddg_search_urls(q, max_results=max_results_per_query)
        if not allow_facebook:
            urls = [u for u in urls if "facebook.com" not in urlparse(u).netloc]
        all_urls.extend(urls)
        progress.progress(i / len(queries))

    # Deduplicate URLs (domain+path)
    cleaned, seen = [], set()
    for u in all_urls:
        key = urlparse(u)._replace(query="", fragment="").geturl()
        if key not in seen:
            cleaned.append(u); seen.add(key)

    st.markdown(
        f"🔗 Candidate pages to scan: **{len(cleaned)}** "
        f"(Seeds: {len(seeds)} + Searched: {max(0, len(cleaned)-len(seeds))})"
    )

    # Scan pages
    results = []
    progress2 = st.progress(0.0)
    for idx, u in enumerate(cleaned, start=1):
        progress2.progress(idx / max(1, len(cleaned)))
        with st.spinner(f"Scanning {u}"):
            hits = harvest_from_url(u, town, user_tz, wk_start, wk_end, enable_ics=enable_ics)
            results.extend(hits)

    # Sort results by start time (unknowns last)
    def sort_key(e):
        s = e["start"]
        return (0, s) if isinstance(s, datetime) else (1, datetime.max)

    results.sort(key=sort_key)

    if not results:
        st.warning("No clear, structured events found. Try adding a nearby larger city or a specific venue keyword, "
                   "or increase 'Max results per query' in the sidebar.")
    else:
        st.success(f"Found {len(results)} event candidates for the weekend.")

        # Present table
        table_rows = []
        for e in results:
            s = e["start"].strftime("%a %b %d %I:%M %p") if isinstance(e["start"], datetime) else ""
            en = e["end"].strftime("%a %b %d %I:%M %p") if isinstance(e["end"], datetime) else ""
            table_rows.append({
                "Event": e["name"],
                "Start": s,
                "End": en,
                "Location": e["location"],
                "Link": e["url"],
                "Source Page": e["source"],
                "Notes": (e["description"][:160] + "…") if len(e["description"]) > 160 else e["description"]
            })

        st.markdown("### Results")
        st.dataframe(table_rows, use_container_width=True)

        # CSV download
        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=table_rows[0].keys())
        writer.writeheader()
        writer.writerows(table_rows)
        st.download_button("⬇️ Download CSV", data=csv_buf.getvalue(),
                           file_name=f"{_slugify(town)}_{state.upper()[:2]}_weekend_events.csv",
                           mime="text/csv")

        # ICS download
        if icalendar:
            cal = icalendar.Calendar()
            cal.add('prodid', '-//Weekend Events Finder//streamlit//')
            cal.add('version', '2.0')
            for e in results:
                if not isinstance(e["start"], datetime):
                    continue
                ve = icalendar.Event()
                ve.add('summary', e["name"])
                ve.add('dtstart', e["start"])
                if isinstance(e["end"], datetime):
                    ve.add('dtend', e["end"])
                if e["location"]:
                    ve.add('location', e["location"])
                desc = e["description"] or ""
                if e["url"]:
                    desc = (desc + "\n" if desc else "") + e["url"]
                ve.add('description', desc)
                cal.add_component(ve)
            ics_bytes = cal.to_ical()
            st.download_button("📆 Add to Calendar (ICS)", data=ics_bytes,
                               file_name=f"{_slugify(town)}_{state.upper()[:2]}_weekend_events.ics",
                               mime="text/calendar")
        else:
            st.info("Install `icalendar` to enable an ICS download.")
