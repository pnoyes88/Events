# app.py
# Streamlit Weekend Events Finder
# --------------------------------
# This app searches local web sources for a given Town, State and extracts events
# for the upcoming weekend (Fri–Sun). It looks for JSON-LD (schema.org Event)
# and .ics calendar links whenever possible, and falls back to simple heuristics.

import streamlit as st
import re
import csv
import io
import json
import math
import itertools
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# Third-party utilities
from dateutil import parser as dateparser
from dateutil import tz

# Search (DuckDuckGo)
try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None

# ICS parsing (optional but helpful)
try:
    import icalendar
except Exception:
    icalendar = None


# ----------------------
# Utility: Weekend range
# ----------------------
def upcoming_weekend_range(user_tz="America/New_York"):
    """Return (start_dt, end_dt) for the upcoming weekend in timezone user_tz.
    Weekend is Friday 00:00 through Sunday 23:59:59."""
    tzinfo = tz.gettz(user_tz)
    now = datetime.now(tzinfo)

    # Monday=0 ... Sunday=6; get days until Friday (4)
    days_until_friday = (4 - now.weekday()) % 7
    # If it's already Fri-Sun, we keep *this* weekend; else go to next Friday.
    # If today is Mon-Thu, we go to the nearest upcoming Fri.
    friday = (now + timedelta(days=days_until_friday)).replace(hour=0, minute=0, second=0, microsecond=0)
    sunday_end = (friday + timedelta(days=2)).replace(hour=23, minute=59, second=59, microsecond=0)
    return friday, sunday_end


# ----------------------
# Search query generator
# ----------------------
def build_queries(town: str, state: str, extra_keywords: str = ""):
    base = f"{town}, {state}".strip()
    ek = extra_keywords.strip()
    q_extra = f" {ek}" if ek else ""

    queries = [
        f"{base} events this weekend{q_extra}",
        f"{base} calendar events{q_extra}",
        f"{base} things to do this weekend{q_extra}",

        # Common local sources
        f"{base} library events{q_extra}",
        f"{base} parks and recreation events{q_extra}",
        f"{base} chamber of commerce events{q_extra}",
        f"{base} town hall events{q_extra}",
        f"site:.gov {base} events OR calendar{q_extra}",
        f"site:.org {base} events OR calendar{q_extra}",

        # Popular aggregators (we only follow links we can fetch)
        f"site:eventbrite.com {base} events{q_extra}",
        f"site:meetup.com {base} events{q_extra}",
        f"site:facebook.com/events {base}{q_extra}",  # FB often blocks scraping; we keep links
        f"site:allevents.in {base}{q_extra}",
        f"site:patch.com {base} events{q_extra}",
    ]
    # Dedup while preserving order
    seen = set()
    out = []
    for q in queries:
        if q not in seen:
            out.append(q)
            seen.add(q)
    return out


# ----------------------
# Web fetching helpers
# ----------------------
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; WeekendEventsBot/1.0; +https://streamlit.io/)",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def fetch_url(url: str, timeout=12):
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        if r.status_code == 200 and r.content:
            return r.text, r.url  # content and final URL
    except Exception:
        return None, url
    return None, url


# ----------------------
# JSON-LD Event extraction
# ----------------------
def _flatten_jsonld(obj):
    # Yield dicts inside lists or singletons
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
        # Some sites concatenate multiple JSON-LD blobs; try to recover arrays
        candidates = []
        try:
            data = json.loads(raw)
            candidates.append(data)
        except Exception:
            # Try to split on '}\s*{'
            parts = re.split(r"}\s*{", raw.strip())
            if len(parts) > 1:
                rebuilt = []
                for i, p in enumerate(parts):
                    if i == 0:
                        rebuilt.append(p + "}")
                    elif i == len(parts)-1:
                        rebuilt.append("{" + p)
                    else:
                        rebuilt.append("{" + p + "}")
                for chunk in rebuilt:
                    try:
                        candidates.append(json.loads(chunk))
                    except Exception:
                        pass

        for data in candidates:
            for node in _flatten_jsonld(data):
                if "@type" in node and _looks_like_event_type(node["@type"]):
                    ev = {}
                    ev["name"] = node.get("name")
                    ev["description"] = node.get("description")
                    ev["url"] = node.get("url") or base_url
                    # Dates
                    start = node.get("startDate") or node.get("start_time")
                    end = node.get("endDate") or node.get("end_time")
                    ev["start"] = start
                    ev["end"] = end
                    # Location
                    loc = node.get("location")
                    if isinstance(loc, dict):
                        ev["location"] = loc.get("name") or loc.get("address") or ""
                    else:
                        ev["location"] = loc
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
    return list(dict.fromkeys(links))  # dedup, preserve order

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
                }
                start = comp.get("dtstart")
                end = comp.get("dtend")
                if start:
                    try:
                        ev["start"] = start.dt.isoformat()
                    except Exception:
                        ev["start"] = str(start.dt)
                if end:
                    try:
                        ev["end"] = end.dt.isoformat()
                    except Exception:
                        ev["end"] = str(end.dt)
                events.append(ev)
    except Exception:
        pass
    return events


# ----------------------
# Heuristic fallback (very light)
# ----------------------
def heuristic_extract_events(html: str, base_url: str, town: str):
    """Very simple fallback: find list items that look like dates + titles."""
    soup = BeautifulSoup(html, "html.parser")
    candidates = []

    # Look for items with 'event' in class/id
    for sel in ["div", "li", "article", "section"]:
        for node in soup.find_all(sel):
            cls = " ".join(node.get("class", [])).lower()
            nid = (node.get("id") or "").lower()
            if "event" in cls or "event" in nid or "calendar" in cls or "calendar" in nid:
                text = " ".join(node.stripped_strings)
                if len(text) > 40:
                    candidates.append((node, text))

    events = []
    # crude date regexes
    date_patterns = [
        r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b.*?\b(?:\d{1,2}:\d{2}\s?(?:AM|PM))?",
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b.*?\b\d{1,2}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",  # ISO date
    ]
    date_re = re.compile("|".join(date_patterns), re.IGNORECASE)

    for node, text in candidates:
        # Find a link inside for URL
        link = node.find("a", href=True)
        url = urljoin(base_url, link["href"]) if link else base_url
        # Extract a line-ish chunk as title
        lines = [s for s in node.stripped_strings]
        if not lines:
            continue
        title = lines[0][:140]
        # Approximate date
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
    out = []
    for e in events:
        s = e["start"]
        # If start is missing, try to parse from description
        if not s and e["description"]:
            s = to_dt(e["description"], wk_start.tzinfo)
            if s:
                e["start"] = s
        if s and (wk_start <= s <= wk_end):
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
                # duckduckgo_search returns dicts with 'href'/'link'
                href = r.get("href") or r.get("link") or r.get("url")
                if href and href.startswith("http"):
                    urls.append(href)
    except Exception:
        pass
    # Dedup domain+path basic
    seen = set()
    res = []
    for u in urls:
        key = urlparse(u)._replace(query="", fragment="").geturl()
        if key not in seen:
            res.append(u)
            seen.add(key)
    return res


def harvest_from_url(url, town, tzname, wk_start, wk_end, enable_ics=True):
    html, final_url = fetch_url(url)
    if not html:
        return []

    # First: JSON-LD Events
    jl = parse_jsonld_events(html, final_url)
    normalized = normalize_events(jl, tzname, final_url)
    weekend_hits = filter_to_weekend(normalized, wk_start, wk_end)

    # Second: ICS (if enabled)
    if enable_ics and icalendar:
        ics_links = discover_ics_links(html, final_url)
        for ics_url in ics_links[:3]:  # limit
            try:
                txt, _ = fetch_url(ics_url)
                if txt:
                    ics_events = parse_ics_events(txt, ics_url)
                    normalized_ics = normalize_events(ics_events, tzname, ics_url)
                    weekend_hits.extend(filter_to_weekend(normalized_ics, wk_start, wk_end))
            except Exception:
                pass

    # Fallback: heuristic scrape if nothing found
    if not weekend_hits:
        crude = heuristic_extract_events(html, final_url, town)
        normalized_crude = normalize_events(crude, tzname, final_url)
        # Let heuristic ones through even if date parsing failed, but we’ll sort later
        # Try to keep only ones with any weekend-ish hint in description if present
        weekendish = []
        weekend_words = re.compile(r"\b(Fri|Friday|Sat|Saturday|Sun|Sunday|weekend)\b", re.I)
        for e in normalized_crude:
            if e["start"] or weekend_words.search(e["description"]):
                weekendish.append(e)
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
    max_results_per_query = st.slider("Max results per query", 3, 15, 6, help="How many links to try from each search query.")
    enable_ics = st.checkbox("Parse .ics calendars when found", value=True)
    allow_facebook = st.checkbox("Include Facebook event links (often non-scrapable)", value=True)
    st.markdown("---")
    st.markdown("**Tip:** Add extra keywords like a neighborhood or local venue to improve precision.")

col1, col2 = st.columns([2, 1])
with col1:
    town = st.text_input("Town", placeholder="e.g., Franklin")
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

    queries = build_queries(town, state, extra)
    all_urls = []
    progress = st.progress(0.0)
    status = st.empty()

    # Gather URLs from queries
    for i, q in enumerate(queries, start=1):
        status.write(f"Searching: _{q}_")
        urls = ddg_search_urls(q, max_results=max_results_per_query)
        # Optionally drop FB if user unchecked
        if not allow_facebook:
            urls = [u for u in urls if "facebook.com" not in urlparse(u).netloc]
        all_urls.extend(urls)
        progress.progress(i / len(queries))

    # Deduplicate URLs (domain+path)
    cleaned = []
    seen = set()
    for u in all_urls:
        key = urlparse(u)._replace(query="", fragment="").geturl()
        if key not in seen:
            cleaned.append(u)
            seen.add(key)

    st.markdown(f"Found **{len(cleaned)}** unique candidate pages to scan.")
    results = []
    scanned = 0
    progress2 = st.progress(0.0)

    for idx, u in enumerate(cleaned, start=1):
        progress2.progress(idx / max(1, len(cleaned)))
        with st.spinner(f"Scanning {u}"):
            hits = harvest_from_url(u, town, user_tz, wk_start, wk_end, enable_ics=enable_ics)
            results.extend(hits)
        scanned += 1

    # Normalize & sort results by start time (unknowns last)
    def sort_key(e):
        s = e["start"]
        return (0, s) if isinstance(s, datetime) else (1, datetime.max)

    results.sort(key=sort_key)

    if not results:
        st.warning("No clear, structured events found. Try adding a nearby city or another keyword (e.g., a venue name), or increase results per query.")
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

        # Downloads
        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=table_rows[0].keys())
        writer.writeheader()
        writer.writerows(table_rows)
        st.download_button("⬇️ Download CSV", data=csv_buf.getvalue(), file_name=f"{town}_{state}_weekend_events.csv", mime="text/csv")

        # Build ICS
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
            st.download_button("📆 Add to Calendar (ICS)", data=ics_bytes, file_name=f"{town}_{state}_weekend_events.ics", mime="text/calendar")
        else:
            st.info("Install `icalendar` to enable an ICS download.")


# -------------
# Footer notes
# -------------
with st.expander("Notes & Tips"):
    st.markdown(
        """
- This app prioritizes structured data (Schema.org JSON-LD) and .ics feeds when present. Those are the most reliable.
- Some sites (e.g., Facebook) block scraping. You can still include links to check manually.
- If results are sparse, try:
  - Adding a nearby larger city (e.g., “Franklin, MA **Boston**”) to widen the net.
  - Adding a venue or keyword (e.g., “common”, “library”, “farmers market”, “concert”).
  - Increasing “Max results per query” in the sidebar.
- Weekend is computed using the timezone you provide (default America/New_York).
"""
    )
