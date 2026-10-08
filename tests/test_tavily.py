import io
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.error
import urllib.request

from mini_harness.app import HarnessConfig
from mini_harness import tavily, web_tools
from mini_harness.web_search import run_search
from mini_harness.llm import ToolCall
from mini_harness.adapters.mock import scripted_plugin
from .support import build_context_with, make_temp_dir, remove_tree


class TavilyTests(unittest.TestCase):
    def result(self, **overrides):
        return {'title': 'Python docs', 'url': 'https://docs.python.org/3/',
                'content': 'Python documentation', **overrides}

    def test_env_configuration_is_optional_and_hidden_from_repr(self):
        config = HarnessConfig.from_env(env={'TAVILY_API_KEY': ' secret-test '}, env_file=None)
        self.assertEqual(config.tavily_api_key, 'secret-test')
        self.assertNotIn('secret-test', repr(config))
        self.assertEqual(HarnessConfig.from_env(env={}, env_file=None).tavily_api_key, '')

    def test_api_is_first_and_semantic_results_keep_provider_order(self):
        download = Mock(side_effect=AssertionError('fallback should not run'))
        items = [self.result(title='第一条中文语义结果'), self.result(url='https://example.com/second', title='Python documentation')]
        with patch.object(tavily, 'search', return_value=items) as search:
            result = run_search('Python 官方文档', 2, download, tavily_api_key='test')
        self.assertEqual(result['status'], 'ok')
        self.assertEqual([x['title'] for x in result['results']], [x['title'] for x in items])
        self.assertEqual([x['provider'] for x in result['attempts']], ['Tavily'])
        search.assert_called_once_with('Python 官方文档', 2, 'test', None, topic='general', period='any')

    def test_missing_key_uses_original_providers(self):
        page = '<li class="b_algo"><h2><a href="https://docs.python.org/">Python docs</a></h2></li>'
        with patch.object(tavily, 'search') as search:
            result = run_search('Python docs', 1, lambda *_: {'text': page})
        search.assert_not_called()
        self.assertEqual(result['results'][0]['provider'], 'Bing Web')

    def test_errors_or_empty_results_fall_back(self):
        page = '<li class="b_algo"><h2><a href="https://docs.python.org/">Python docs</a></h2></li>'
        for effect in (tavily.TavilyError('Tavily HTTP 401：密钥无效'), []):
            with self.subTest(effect=effect), patch.object(tavily, 'search') as search:
                if isinstance(effect, Exception): search.side_effect = effect
                else: search.return_value = effect
                result = run_search('Python docs', 1, lambda *_: {'text': page}, tavily_api_key='test')
                self.assertEqual(result['status'], 'ok')
                self.assertEqual(result['results'][0]['provider'], 'Bing Web')
                self.assertEqual(result['attempts'][0]['provider'], 'Tavily')

    def test_news_dates_invalid_links_and_duplicates_are_filtered(self):
        now = datetime.now(timezone.utc)
        good = self.result(published_date=now.isoformat())
        items = [None, {'url': 'javascript:alert(1)'}, good, good,
                 self.result(url='https://example.com/old', published_date=(now-timedelta(days=8)).isoformat()),
                 self.result(url='https://example.com/no-date')]
        with patch.object(tavily, 'search', return_value=items):
            result = run_search('Python news', 1, Mock(), topic='news', time_range='week', tavily_api_key='test')
        self.assertEqual(len(result['results']), 1)
        self.assertEqual(result['results'][0]['provider'], 'Tavily')
        self.assertEqual(result['attempts'][0]['filtered']['missing_date'], 1)
        self.assertEqual(result['attempts'][0]['filtered']['out_of_range'], 1)

    def test_transport_sends_bounded_basic_search_and_bearer_header(self):
        opener = Mock()
        opener.open.return_value = io.BytesIO(json.dumps({'results': [self.result()]}).encode())
        with patch.object(tavily.urllib.request, 'build_opener', return_value=opener):
            result = tavily.search('Python', 3, 'secret-test', topic='news', period='week')
        request = opener.open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, 'https://api.tavily.com/search')
        self.assertEqual(request.get_header('Authorization'), 'Bearer secret-test')
        self.assertNotIn('secret-test', request.data.decode())
        self.assertEqual(payload['search_depth'], 'basic')
        self.assertEqual(payload['time_range'], 'week')
        self.assertTrue(payload['include_published_date'])
        self.assertEqual(payload['max_results'], 3)
        self.assertEqual(len(result), 1)

    def test_http_errors_never_echo_response_or_credential(self):
        for status in (401, 403, 429, 432, 433, 500):
            with self.subTest(status=status):
                error = urllib.error.HTTPError('https://api.tavily.com/search', status, 'secret-test', {}, io.BytesIO(b'secret-test'))
                opener = Mock(); opener.open.side_effect = error
                with patch.object(tavily.urllib.request, 'build_opener', return_value=opener):
                    with self.assertRaises(tavily.TavilyError) as raised:
                        tavily.search('x', 1, 'secret-test')
                self.assertIn(str(status), str(raised.exception))
                self.assertNotIn('secret-test', str(raised.exception))

    def test_bad_json_oversize_timeout_and_cancellation(self):
        for body in (b'not-json secret-test', b'{}', b'x' * (2*1024*1024+1)):
            opener = Mock(); opener.open.return_value = io.BytesIO(body)
            with patch.object(tavily.urllib.request, 'build_opener', return_value=opener):
                with self.assertRaises(tavily.TavilyError) as raised:
                    tavily.search('x', 1, 'secret-test')
                self.assertNotIn('secret-test', str(raised.exception))
        with patch.object(tavily.urllib.request, 'build_opener') as build:
            with self.assertRaisesRegex(tavily.TavilyError, '取消'):
                tavily.search('x', 1, 'test', SimpleNamespace(cancelled=True))
            build.assert_not_called()
        opener = Mock(); opener.open.side_effect = TimeoutError()
        with patch.object(tavily.urllib.request, 'build_opener', return_value=opener):
            with self.assertRaisesRegex(tavily.TavilyError, '超时'):
                tavily.search('x', 1, 'test')

    def test_redirect_does_not_forward_authorization(self):
        request = urllib.request.Request('https://api.tavily.com/search', headers={'Authorization': 'Bearer secret-test'})
        with self.assertRaisesRegex(tavily.TavilyError, '重定向'):
            tavily.NoRedirect().redirect_request(request, None, 302, 'redirect', {}, 'https://other.example/')


class TavilyWiringTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_plugin_passes_key_to_search_without_exposing_it(self):
        root = make_temp_dir()
        ctx = build_context_with(scripted_plugin([]), root, tavily_api_key='secret-test')
        try:
            with patch.object(tavily, 'search', return_value=[{'url': 'https://docs.python.org/', 'title': 'Python', 'content': 'docs'}]) as search:
                result = await ctx.tools.execute(ToolCall('search', 'web_search', {'queries': ['Python'], 'count': 1}), ctx.sessions.create(), root)
            self.assertFalse(result.is_error)
            self.assertEqual(search.call_args.args[2], 'secret-test')
            self.assertNotIn('secret-test', result.content)
            self.assertEqual(json.loads(result.content)['queries'][0]['results'][0]['provider'], 'Tavily')
        finally:
            ctx.dispose()
            await remove_tree(root)
