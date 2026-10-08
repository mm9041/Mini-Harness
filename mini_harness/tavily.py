"""Bounded Tavily Search API transport; credentials never enter diagnostics."""
import json
import socket
import time
import urllib.error
import urllib.request


class TavilyError(RuntimeError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the bearer credential to a redirected destination.
        raise TavilyError("Tavily 返回重定向，已拒绝转发凭据")


def search(query, count, api_key, cancellation=None, *, topic="general", period="any"):
    def check():
        if cancellation and cancellation.cancelled:
            raise TavilyError("网络请求已取消")
    check()
    payload = {"query": query, "max_results": count, "topic": topic,
               "search_depth": "basic", "include_answer": False,
               "include_raw_content": False, "include_images": False,
               "include_published_date": True}
    if period != "any":
        payload["time_range"] = period
    request = urllib.request.Request(
        "https://api.tavily.com/search", data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"})
    try:
        started = time.monotonic()
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=15) as response:
            chunks, size = [], 0
            while True:
                check()
                if time.monotonic() - started > 30:
                    raise TavilyError("Tavily 请求超时")
                part = response.read1(min(65536, 2 * 1024 * 1024 + 1 - size))
                if not part:
                    break
                chunks.append(part)
                size += len(part)
                if size > 2 * 1024 * 1024:
                    raise TavilyError("Tavily 响应超过 2 MB")
        check()
        data = json.loads(b"".join(chunks))
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise TavilyError("Tavily 返回无效的搜索结果结构")
        return data["results"]
    except urllib.error.HTTPError as exc:
        exc.close()
        reason = {401: "密钥无效", 403: "访问被拒绝", 429: "请求限流", 432: "密钥额度不足",
                  433: "账户额度不足"}.get(exc.code, "搜索服务错误")
        raise TavilyError(f"Tavily HTTP {exc.code}：{reason}") from None
    except (TimeoutError, socket.timeout):
        raise TavilyError("Tavily 请求超时") from None
    except urllib.error.URLError:
        raise TavilyError("Tavily 网络连接失败") from None
    except (ValueError, UnicodeError):
        raise TavilyError("Tavily 返回无效 JSON") from None
