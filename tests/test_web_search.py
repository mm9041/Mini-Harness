import asyncio
import base64
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import json
import unittest
from unittest.mock import patch
import urllib.error
import urllib.parse
from xml.sax.saxutils import escape

from mini_harness import web_tools
from mini_harness.web_search import SearchHTML, parse_date, result_url, run_search, query_terms, safe_link
from mini_harness.adapters.mock import scripted_plugin
from mini_harness.llm import ToolCall
from .support import build_context_with, make_temp_dir, remove_tree


def rss(items):
    return '<rss><channel>' + ''.join(
        '<item><title>' + escape(title) + '</title><link>' + escape(url) + '</link>'
        '<description>' + escape(title) + '</description><pubDate>' + escape(date) + '</pubDate>'
        '<source url="https://publisher.example">Publisher</source></item>'
        for title, url, date in items) + '</channel></rss>'


def html(title, url, provider='Bing Web'):
    if provider == 'Bing Web':
        return f'<ol><li class="b_algo"><h2><a href="{url}">{title}</a></h2><p>{title}</p></li></ol>'
    return f'<div class="result"><div><a class="result__a" href="{url}">{title}</a><a class="result__snippet">{title}</a></div></div>'


class SearchProviderTests(unittest.TestCase):
    def test_news_noise_is_removed_without_erasing_real_single_character_topics(self):
        for query, expected in (('C news',['c']), ('R news',['r']), ('X news',['x']),
                                ('美 新闻',['美']), ('日 新闻',['日']), ("NASA's news",['nasa'])):
            with self.subTest(query=query):
                self.assertEqual(query_terms(query, 'news'), expected)
        self.assertEqual(query_terms('Python 3.13 有哪些新特性','news'),
                         ['python','3','13','有哪','哪些','些新','新特','特性'])
        self.assertTrue({'量子','子计','计算'}.issubset(query_terms('最近的量子计算新闻','news')))

    def test_specific_single_letter_news_does_not_use_headlines(self):
        seen = []
        def download(url, cancellation):
            seen.append(url)
            return {'text':rss([('C language update','https://publisher.example/c',format_datetime(datetime.now(timezone.utc)))])}
        result = run_search('C news', 1, download)
        self.assertIn('bing.com/news/search', seen[0])
        self.assertNotIn('search_scope', result)
        self.assertEqual(result['status'], 'ok')

    def test_safe_link_rejects_bad_ports_and_accepts_valid_absolute_or_relative_links(self):
        for url in ('https://ex.com:abc/', 'https://ex.com:65536/', 'https://ex.com:-1/',
                    'https://[::1', 'javascript:alert(1)', 'https://u:p@example.com/x',
                    'https://ex.com/' + 'a'*4001):
            with self.subTest(url=url):
                self.assertEqual(safe_link(url), '')
        self.assertEqual(safe_link('https://ex.com:8443/a#part'), 'https://ex.com:8443/a')
        self.assertEqual(safe_link('/story', 'https://ex.com:8443/news'), 'https://ex.com:8443/story')
        self.assertEqual(safe_link('/story', 'https://ex.com:abc/news'), '')

    def test_bing_html_decodes_result_links_and_nested_titles(self):
        dest = 'https://docs.python.org/3/'
        encoded = base64.urlsafe_b64encode(dest.encode()).decode().rstrip('=')
        parser = SearchHTML('Bing Web', 'https://www.bing.com/search')
        parser.feed(html('Python <strong>documentation</strong>', '/ck/a?u=a1' + encoded))
        self.assertEqual(parser.results[0]['url'], dest)
        self.assertEqual(parser.results[0]['title'], 'Python documentation')
        self.assertEqual(result_url('/l/?uddg=https%3A%2F%2Fexample.com%2Farticle', 'https://duckduckgo.com'), 'https://example.com/article')
        self.assertEqual(result_url('/ck/a?u=a1***', 'https://www.bing.com'), '')
        self.assertEqual(result_url('javascript:alert(1)', 'https://example.com'), '')

    def test_irrelevant_primary_and_rss_fall_back_without_returning_garbage(self):
        calls = []
        def download(url, cancellation):
            calls.append(url)
            if 'format=rss' in url:
                text = rss([('MINI cars', 'https://cars.example/', '')])
            elif 'duckduckgo' in url:
                text = html('mini harness agent GitHub', 'https://github.com/example/mini-harness', 'DuckDuckGo HTML')
            else:
                text = html('MINI cars', 'https://cars.example/')
            return {'url': url, 'text': text}
        result = run_search('mini harness agent GitHub', 1, download)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['attempts'][0]['filtered']['irrelevant'], 1)
        self.assertEqual(result['results'][0]['provider'], 'DuckDuckGo HTML')
        self.assertNotIn('cars.example', json.dumps(result))

    def test_news_filters_stale_future_missing_dates_and_irrelevant_items(self):
        now = datetime.now(timezone.utc)
        date = lambda delta: format_datetime(now + delta)
        wrapped = 'https://www.bing.com/news/apiclick.aspx?' + urllib.parse.urlencode({'url':'https://publisher.example/python'})
        items = [('Python news released', wrapped, date(timedelta(hours=-2))),
                 ('Python archive', 'https://publisher.example/old', date(timedelta(days=-5))),
                 ('Python future', 'https://publisher.example/future', date(timedelta(days=1))),
                 ('Python unknown', 'https://publisher.example/unknown', ''),
                 ('HSBC banking', 'https://bank.example/a', date(timedelta(hours=-1)))]
        result = run_search('Python news', 1, lambda u,c: {'text':rss(items)}, topic='news')
        self.assertEqual(result['time_range'], 'day')
        self.assertEqual(len(result['results']), 1)
        self.assertEqual(result['results'][0]['url'], 'https://publisher.example/python')
        self.assertEqual(result['results'][0]['publisher'], 'Publisher')
        self.assertEqual(result['attempts'][0]['filtered'], {'irrelevant':1,'missing_date':1,'out_of_range':2})

    def test_broad_news_uses_headlines_and_keeps_aggregator_label(self):
        seen = []
        def download(url,c):
            seen.append(url)
            return {'text':rss([('World event', 'https://news.google.com/rss/articles/abc', format_datetime(datetime.now(timezone.utc)))])}
        for query in ('国际新闻 今日最新', '最近的国际新闻', '国际新闻 今日的最新',
                      "today's world news", 'today’s world news', '最近的国际新闻呢'):
            with self.subTest(query=query):
                seen.clear()
                result = run_search(query, 1, download)
                self.assertIn('/headlines/section/topic/WORLD', seen[0])
                self.assertNotIn('q=', seen[0])
                self.assertEqual(result['topic'], 'news')
                self.assertEqual(result['results'][0]['link_kind'], 'aggregator')
                self.assertIn('search_scope', result)

    def test_partial_success_preserves_source_failure(self):
        def download(url,c):
            if 'bing.com' in url:
                return {'text':rss([('Python news', 'https://publisher.example/a', format_datetime(datetime.now(timezone.utc)))])}
            raise urllib.error.HTTPError(url, 503, 'unavailable', {}, None)
        result = run_search('Python news', 3, download)
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(len(result['results']), 1)
        self.assertNotIn('error', result)
        self.assertIn('503', result['attempts'][1]['error'])

    def test_all_irrelevant_or_blocked_is_unavailable(self):
        result = run_search('Python programming', 2, lambda u,c: {'text':html('bank account', 'https://bank.example/')})
        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(result['results'], [])
        self.assertIn('SEARCH_UNAVAILABLE', result['error'])

    def test_explicit_historical_range_and_news_topic_override(self):
        item = ('Python release', 'https://publisher.example/a', format_datetime(datetime.now(timezone.utc)-timedelta(days=3)))
        result = run_search('Python', 1, lambda u,c:{'text':rss([item])}, topic='news', time_range='week')
        self.assertEqual(result['status'], 'ok')
        result = run_search('Python news', 1, lambda u,c:{'text':html('Python news documentation','https://python.example')}, topic='general', time_range='day')
        self.assertEqual(result['topic'], 'general')
        self.assertIn('warning', result)

    def test_cancellation_does_not_start_a_fallback(self):
        class Cancellation:
            cancelled = False
        cancellation = Cancellation()
        calls = []
        def download(url,c):
            calls.append(url)
            c.cancelled = True
            raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError, '取消'):
            run_search('Python', 3, download, cancellation)
        self.assertEqual(len(calls), 1)

    def test_date_parsing_does_not_invent_a_timezone(self):
        self.assertIsNone(parse_date('2026-01-01T12:00:00'))
        self.assertIsNone(parse_date('yesterday'))
        self.assertEqual(parse_date('2026-01-01T12:00:00+08:00').hour, 4)


class SearchToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = make_temp_dir()
        self.ctx = build_context_with(scripted_plugin([]), self.root, permission_preset='workspace-write')
        self.session = self.ctx.sessions.create()

    async def asyncTearDown(self):
        self.ctx.dispose()
        await remove_tree(self.root)

    async def call(self, name, args):
        return await self.ctx.tools.execute(ToolCall('test', name, args), self.session, self.root)

    async def test_no_usable_search_is_tool_error_in_every_permission_mode(self):
        with patch.object(web_tools, 'download', return_value={'text':'<html>captcha</html>'}):
            for mode in ('read-only','workspace-write','danger-full-access'):
                self.ctx.permissions.set(mode,self.session)
                result = await self.call('web_search', {'queries':['Python'], 'topic':'news', 'time_range':'week'})
                self.assertTrue(result.is_error)
                self.assertIn('SEARCH_UNAVAILABLE', result.content)
        self.assertEqual(self.ctx.approval.history, [])

    async def test_search_options_validate_and_reach_provider(self):
        for options in ({'topic':'other'}, {'time_range':[]}, {'time_range':'yesterday'}, {'topic':False}):
            with patch.object(web_tools, 'download') as download:
                self.assertTrue((await self.call('web_search', {'queries':['Python'], **options})).is_error)
                download.assert_not_called()
        with patch.object(web_tools, 'search_one', return_value={'results':[]}) as search:
            await self.call('web_search', {'queries':['Python'], 'topic':'news', 'time_range':'week'})
            self.assertEqual(search.call_args.kwargs, {'topic':'news','time_range':'week'})

    async def test_fetch_keeps_article_links_and_explicit_dates_separate_from_fetch_time(self):
        text = ('<title>News page</title><meta property="og:type" content="article">'
                '<meta property="article:published_time" content="2026-01-01T12:00:00+08:00">'
                '<meta property="article:modified_time" content="2026-01-02T12:00:00+08:00">'
                '<p>' + 'text ' * 250 + '</p><a href="/article/1">A complete article title</a>'
                '<a href="/article/1#section">Duplicate title</a>'
                '<a href="javascript:alert(1)">Unsafe title</a><a href="https://user:pass@example.com">Credentials</a>'
                '<script><a href="https://evil.example">Invisible link</a></script>')
        with patch.object(web_tools,'download', return_value={'url':'https://example.com/news','content_type':'text/html','text':text}):
            result=await self.call('web_fetch', {'url':'https://example.com/news','max_chars':1000})
        data=json.loads(result.content)
        self.assertTrue(data['truncated'])
        self.assertEqual(data['links'],[{'title':'A complete article title','url':'https://example.com/article/1'}])
        self.assertEqual(data['published_at'],'2026-01-01T04:00:00+00:00')
        self.assertNotEqual(data['fetched_at'],data['published_at'])
        self.assertEqual(data['page_kind'],'article')

    async def test_homepage_times_are_not_treated_as_article_publication_dates(self):
        parser=web_tools.PageText('https://example.com')
        parser.feed('<time datetime="2026-10-04T00:00:00Z">Today</time><article>Headline</article>')
        self.assertIsNone(parser.evidence()['published_at'])
        self.assertEqual(parser.evidence()['page_kind'],'page')

    async def test_jsonld_dates_are_read_without_executing_or_exposing_scripts(self):
        parser=web_tools.PageText('https://example.com/article')
        parser.feed('<script type="application/ld+json">{"@graph":[{"@type":"NewsArticle","datePublished":"2026-01-01T12:00:00Z"}]}</script><p>Visible text</p>')
        self.assertEqual(parser.evidence()['published_at'],'2026-01-01T12:00:00+00:00')
        self.assertEqual(parser.evidence()['page_kind'],'article')
        self.assertEqual(parser.result()[1],'Visible text')
        multiple=web_tools.PageText('https://example.com')
        multiple.feed('<script type="application/ld+json">[{"@type":"NewsArticle","datePublished":"2026-01-01T12:00:00Z"},{"@type":"NewsArticle","datePublished":"2026-01-02T12:00:00Z"}]</script>')
        self.assertIsNone(multiple.evidence()['published_at'])
        self.assertEqual(multiple.evidence()['page_kind'],'page')


if __name__ == '__main__':
    unittest.main()
