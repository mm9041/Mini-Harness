"""Bounded public search providers and conservative relevance/freshness checks."""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from html.parser import HTMLParser
import math
import re
import urllib.parse
import xml.etree.ElementTree as ET


PERIODS = {"day": 1, "week": 7, "month": 31, "year": 366, "any": None}
NEWS_WORDS = r"新闻|新聞|头条|頭條|热点|熱點|\b(?:news|headlines|breaking)\b"
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


def clean_text(value):
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]*>", " ", value))).strip()


def safe_link(value, base=""):
    if not value or len(value) > 4000 or any(ord(c) < 32 for c in value):
        return ""
    try:
        url = urllib.parse.urljoin(base, value)
        p = urllib.parse.urlsplit(url)
        if p.scheme not in ("https", "http") or not p.hostname or p.username or p.password:
            return ""
        _ = p.port  # urlsplit defers malformed/out-of-range port validation until access.
        return urllib.parse.urldefrag(url)[0]
    except ValueError:
        return ""


def result_url(value, base):
    """Decode documented URL wrappers, never execute page scripts or make extra requests."""
    url = safe_link(value, base)
    if not url:
        return ""
    p = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qs(p.query)
    if p.hostname in ("www.bing.com", "bing.com"):
        if p.path == "/news/apiclick.aspx":
            return safe_link(query.get("url", [""])[0])
        if p.path == "/ck/a":
            encoded = query.get("u", [""])[0]
            if encoded.startswith("a1"):
                try:
                    data = encoded[2:]
                    return safe_link(base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8"))
                except (ValueError, UnicodeError):
                    return ""
            return ""
    if p.hostname in ("duckduckgo.com", "html.duckduckgo.com") and "uddg" in query:
        return safe_link(query["uddg"][0])
    return url


class SearchHTML(HTMLParser):
    def __init__(self, provider, base):
        super().__init__(convert_charrefs=True)
        self.provider, self.base = provider, base
        self.stack, self.results = [], []
        self.current = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        start = (tag == "li" and "b_algo" in classes) if self.provider == "Bing Web" else (tag == "div" and "result" in classes)
        if start and self.current is None:
            self.current = {"title": [], "snippet": [], "url": "", "depth": len(self.stack)}
        role = ""
        if self.current is not None:
            if tag == "h2" or "result__a" in classes:
                role = "title"
            elif tag == "p" or "result__snippet" in classes:
                role = "snippet"
            if tag == "a" and (role == "title" or any(r == "title" for _, r in self.stack)):
                self.current["url"] = result_url(attrs.get("href", ""), self.base)
        if tag not in VOID:
            self.stack.append((tag, role))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break
        if self.current is not None and len(self.stack) <= self.current["depth"]:
            row = self.current
            if row["url"]:
                self.results.append({"title": clean_text("".join(row["title"])), "url": row["url"],
                                     "snippet": clean_text("".join(row["snippet"]))[:1500], "published_at": None})
            self.current = None

    def handle_data(self, text):
        if self.current is not None:
            roles = {role for _, role in self.stack}
            for role in ("title", "snippet"):
                if role in roles:
                    self.current[role].append(text)


def parse_date(value):
    if not value:
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            result = parsedate_to_datetime(value)
        except (ValueError, TypeError, OverflowError):
            return None
    # No timezone means the age cannot be established reliably.
    return result.astimezone(timezone.utc) if result.tzinfo else None


def rss_results(text, base):
    root = ET.fromstring(text)
    if root.tag != "rss":
        raise ValueError("服务没有返回 RSS（可能重定向到首页或验证页面）")
    rows = []
    for item in root.findall("./channel/item")[:100]:
        url = result_url(item.findtext("link", ""), base)
        if not url:
            continue
        source = next((child for child in item if child.tag.rsplit("}", 1)[-1].lower() == "source"), None)
        date = parse_date(item.findtext("pubDate", ""))
        rows.append({"title": clean_text(item.findtext("title", "")), "url": url,
                     "snippet": clean_text(item.findtext("description", ""))[:1500],
                     "published_at": date.isoformat() if date else None,
                     "publisher": (source.text or "")[:200] if source is not None else "",
                     "publisher_url": safe_link(source.get("url", "")) if source is not None else "",
                     "link_kind": "aggregator" if urllib.parse.urlsplit(url).hostname == "news.google.com" else "source"})
    return rows


def query_terms(query, topic):
    # Strip possessives before removing news/time words: today's must not leave
    # an orphan 's'. Do not discard all single letters (C/R/X) or country initials.
    value = re.sub(r"\b([a-z]+)['’]s\b", r"\1", query.lower())
    if topic == "news":
        value = re.sub(NEWS_WORDS + r"|今日|今天|最近|最新|热点|熱點|头条|頭條|实时|即時|\b(?:today|latest|recent|top|current)\b", " ", value)
        # Broad geographic news requests map to a headlines feed, not a literal phrase search.
        value = re.sub(r"国际|國際|国内|國內|全球|世界|\b(?:world|international|national|headlines)\b", " ", value)
    value = value.replace("官方", " official ").replace("文档", " documentation ")
    words = re.findall(r"[a-z0-9]+|[\u3400-\u9fff]+", value)
    terms = []
    for word in words:
        if word in {"the", "a", "an", "of", "for", "and", "in", "on", "to", "is", "with",
                    "的", "了", "是", "在", "吗", "嗎", "呢", "啊"}:
            continue
        if re.search(r"[\u3400-\u9fff]", word) and len(word) > 1:
            terms.extend(word[i:i+2] for i in range(len(word)-1))
        else:
            terms.append(word)
    return list(dict.fromkeys(terms))


def matched_terms(row, terms):
    text = (row["title"] + " " + row["snippet"] + " " + urllib.parse.unquote(row["url"])).lower()
    return sum(bool(re.search(r"\b" + re.escape(t) + r"\b", text)) if t.isascii() else t in text for t in terms)


def relevant(row, terms):
    return not terms or matched_terms(row, terms) >= max(1, math.ceil(len(terms) / 2))


def provider_request(provider, query, topic, period, broad):
    chinese = bool(re.search(r"[\u3400-\u9fff]", query))
    if provider == "Bing Web":
        params = {"q": query, "mkt": "zh-CN" if chinese else "en-US"}
        if period != "any":
            params["filters"] = 'ex1:"' + {"day": "ez1", "week": "ez2", "month": "ez3", "year": "ez5"}[period] + '"'
        return "https://www.bing.com/search?" + urllib.parse.urlencode(params)
    if provider == "DuckDuckGo HTML":
        params = {"q": query}
        if period != "any":
            params["df"] = {"day": "d", "week": "w", "month": "m", "year": "y"}[period]
        return "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode(params)
    if provider == "Bing RSS":
        return "https://www.bing.com/search?" + urllib.parse.urlencode({"q": query, "format": "rss"})
    if provider == "Bing News":
        # zh-CN news requests can be redirected to the Bing homepage. en-US
        # still accepts Chinese queries and provides source URLs and pubDate.
        return "https://www.bing.com/news/search?" + urllib.parse.urlencode({"q": query, "format": "rss", "mkt": "en-US"})
    params = {"hl": "zh-CN" if chinese else "en-US", "gl": "CN" if chinese else "US",
              "ceid": "CN:zh-Hans" if chinese else "US:en"}
    endpoint = "https://news.google.com/rss"
    if broad:
        if re.search(r"国际|國際|全球|世界|\b(?:world|international)\b", query, re.I):
            endpoint += "/headlines/section/topic/WORLD"
        elif re.search(r"国内|國內|\bnational\b", query, re.I):
            endpoint += "/headlines/section/topic/NATION"
    else:
        endpoint += "/search"
        params["q"] = query + (f" when:{PERIODS[period]}d" if period != "any" else "")
    return endpoint + "?" + urllib.parse.urlencode(params)


def run_search(query, count, download, cancellation=None, *, topic="auto", time_range=None, tavily_api_key=""):
    if topic == "auto":
        topic = "news" if re.search(NEWS_WORDS, query, re.I) else "general"
    period = time_range or ("day" if topic == "news" else "any")
    terms = query_terms(query, topic)
    broad = topic == "news" and not terms
    general = ("Bing Web", "Bing RSS", "DuckDuckGo HTML") if period == "any" else ("Bing Web", "DuckDuckGo HTML")
    providers = ("Google News", "Bing News") if broad else (("Bing News", "Google News") if topic == "news" else general)
    if tavily_api_key:
        providers = ("Tavily",) + providers
    now = datetime.now(timezone.utc)
    rows, attempts, seen = [], [], set()
    for provider in providers:
        if cancellation and cancellation.cancelled:
            raise RuntimeError("网络请求已取消")
        try:
            if provider == "Tavily":
                from .tavily import search
                raw = search(query, count, tavily_api_key, cancellation, topic=topic, period=period)
                candidates = []
                for item in raw:
                    if not isinstance(item, dict) or not isinstance(item.get("url"), str):
                        continue
                    link = safe_link(item["url"])
                    if not link:
                        continue
                    stamp = item.get("published_date")
                    date = parse_date(stamp) if isinstance(stamp, str) else None
                    candidates.append({"title": clean_text(str(item.get("title") or "")), "url": link,
                                       "snippet": clean_text(str(item.get("content") or ""))[:1500],
                                       "published_at": date.isoformat() if date else None,
                                       "publisher": urllib.parse.urlsplit(link).hostname, "link_kind": "source"})
            else:
                url = provider_request(provider, query, topic, period, broad)
                page = download(url, cancellation)
                if topic == "news" or provider == "Bing RSS":
                    candidates = rss_results(page["text"], page.get("url", url))
                else:
                    parser = SearchHTML(provider, page.get("url", url))
                    parser.feed(page["text"])
                    candidates = parser.results
                    if not candidates:
                        raise ValueError("搜索页没有可解析结果（可能是验证页面、无结果或页面结构已变）")
            if cancellation and cancellation.cancelled:
                raise RuntimeError("网络请求已取消")
            filtered = {"irrelevant": 0, "missing_date": 0, "out_of_range": 0}
            accepted = 0
            for row in candidates:
                # API semantic ranking can match across languages without literal
                # keyword overlap. Scraped fallback results still need our filter.
                if provider != "Tavily" and not relevant(row, terms):
                    filtered["irrelevant"] += 1
                    continue
                if topic == "news":
                    date = parse_date(row.get("published_at"))
                    if date is None:
                        filtered["missing_date"] += 1
                        continue
                    if date > now + timedelta(hours=1) or (period != "any" and now - date > timedelta(days=PERIODS[period])):
                        filtered["out_of_range"] += 1
                        continue
                if row["url"] not in seen:
                    seen.add(row["url"])
                    rows.append({**row, "provider": provider})
                    accepted += 1
            attempts.append({"provider": provider, "accepted": accepted, "filtered": filtered})
        except Exception as exc:
            if cancellation and cancellation.cancelled:
                raise RuntimeError("网络请求已取消") from exc
            attempts.append({"provider": provider, "error": str(exc)})
        if len(rows) >= count:
            break
    if topic == "news":
        rows.sort(key=lambda r: r["published_at"], reverse=True)
    else:
        rows.sort(key=lambda r: (r["provider"] != "Tavily", -matched_terms(r, terms) if r["provider"] != "Tavily" else 0))
    result = {"query": query, "topic": topic, "time_range": period, "results": rows[:count],
              "status": "ok" if len(rows) >= count else ("partial" if rows else "unavailable"),
              "attempts": attempts, "checked_at": now.isoformat(),
              "verification": "Tavily 结果沿用搜索源相关性排序，备用来源做关键词筛选；新闻在本地检查日期。未核实报道事实，须读取原文并核对来源。"}
    if broad:
        result["search_scope"] = "宽泛新闻查询优先使用搜索 API（如已配置），备用来源使用新闻头条/地区频道；按发布时间过滤。"
    if topic == "general" and period != "any":
        result["warning"] = "时间范围已传给搜索源；普通网页未提供可靠发布时间，无法在本地确认时效。"
    if not rows:
        result["error"] = "SEARCH_UNAVAILABLE: 搜索源失败或没有通过相关性/日期筛选的结果；不代表互联网上没有相关内容。"
    return result
