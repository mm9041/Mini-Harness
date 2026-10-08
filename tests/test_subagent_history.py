import asyncio
from pathlib import Path
import unittest
import shutil
import subprocess

from mini_harness.adapters.mock import ScriptedAdapter
from mini_harness.history import ConversationHistory
from mini_harness.webui import transcript_of
from .support import adapter_plugin_for, build_context_with, make_temp_dir, remove_tree
from .test_webui import post_json


@unittest.skipUnless(shutil.which('node'), 'Node.js required')
class SubagentLogFrontendTests(unittest.TestCase):
    def test_log_button_reads_without_switching_and_renders_plain_text(self):
        source = (Path(__file__).resolve().parents[1] /
                  'mini_harness/webui_static/index.html').read_text(encoding='utf-8')
        handler = source[source.index('function addSubagentLog('):source.index('function addNotice(')]
        script = r"""
const assert = require('node:assert/strict');
const el = (tag, cls, text) => ({tag, textContent:text, children:[], open:false,
  appendChild(n){this.children.push(n);}, showModal(){this.open=true;}, close(){this.open=false;}});
const nodes = new Map();
const $ = id => {if (!nodes.has(id)) nodes.set(id, el('div')); return nodes.get(id);};
let sessionId='parent';
const transcript=el('div'), t=s=>s, trackTurnNode=n=>n, scroll=()=>{};
let post;
""" + handler + r"""
(async()=>{
  const raw='<script>alert(1)</script>';
  post=async(path, body)=>{
    assert.equal(path,'/api/subagent/log');
    assert.deepEqual(body,{session:'parent',child_id:'child'});
    return {title:'计算',content:raw};
  };
  addSubagentLog({saved:true,title:'计算',child_id:'child'});
  const button=transcript.children[0].children[0];
  await button.onclick();
  assert.equal(sessionId,'parent');
  assert.equal($('subagentLogContent').textContent,raw);
  assert.equal($('subagentLogDialog').open,true);
  $('closeSubagentLog').onclick();
  assert.equal($('subagentLogDialog').open,false);
  addSubagentLog({saved:false,title:'失败'});
  assert.equal(transcript.children[1].children[0].tag,'span');
  post=async()=>null;
  await button.onclick();
  assert.equal(button.disabled,false);
  assert.equal($('subagentLogDialog').open,false);
  post=async()=>{sessionId='other';return {title:'old',content:'old'};};
  await button.onclick();
  assert.equal($('subagentLogDialog').open,false);
})().catch(e=>{console.error(e);process.exitCode=1;});
"""
        result = subprocess.run(['node', '-e', script], capture_output=True,
                                encoding='utf-8', timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        # Compile the entire shipped script too, without executing browser globals.
        full_script = source.split('<script>', 1)[1].rsplit('</script>', 1)[0]
        result = subprocess.run(['node', '--check'], input=full_script,
                                capture_output=True, encoding='utf-8', timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


class SubagentHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = make_temp_dir()
        self.ctx = build_context_with(
            adapter_plugin_for(ScriptedAdapter([{'text': '5556'}])), self.root,
            session_root=self.root / 'sessions', compaction=False)
        self.ui = self.ctx.webui
        self.ui.attach()
        self.ui.bind_loop(asyncio.get_running_loop())
        self.ui.autosave = True
        self.parent = self.ui.session
        self.parent.append('user/message', text='委派计算')
        self.result = await self.ctx.subagents.run(
            '12345-6789', parent_session=self.parent, title='计算')
        self.ctx.sessions.save(self.parent)

    async def asyncTearDown(self):
        await self.ui.shutdown()
        self.ui.stop()
        self.ctx.dispose()
        await remove_tree(self.root)

    async def test_restart_and_read_only_log_preserve_parent_and_child(self):
        path = Path(self.result.session_path)
        before = path.read_bytes(), path.stat().st_mtime_ns
        history = ConversationHistory(self.ctx.sessions.root)
        self.assertEqual([r['title'] for r in history.list()], ['委派计算'])
        # Reopen the parent from disk to exercise durable lineage, not live hooks.
        restored = self.ctx.sessions.open(self.parent.source_path)
        self.ui.attach(restored)
        workers = len(self.ui._workers)
        events = len(restored.events)
        entry, = [m for m in transcript_of(restored) if m['kind'] == 'subagent-log']
        self.assertTrue(entry['saved'])
        self.assertEqual(entry['child_id'], self.result.session_id)
        data = self.ui.subagent_log(entry['child_id'], restored.id)
        self.assertIn('5556', data['content'])
        self.assertIs(self.ui.session, restored)
        self.assertEqual(len(self.ui._workers), workers)
        self.assertEqual(len(restored.events), events)
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)
        self.assertNotIn(self.result.session_id, self.ctx.sessions._live)
        self.assertEqual(len(self.ui.history_items()), 1)

    async def test_child_opened_explicitly_stays_out_of_sidebar(self):
        key = self.ui.history.key(Path(self.result.session_path))
        self.ui.open_history(key)
        self.assertEqual(len(self.ui.history_items()), 1)
        self.assertEqual(self.ui.history_items()[0]['title'], '委派计算')

    async def test_missing_unsaved_and_unrelated_logs_report_errors(self):
        for child_id in ('unknown', str(self.parent.source_path), '../outside'):
            with self.assertRaisesRegex(ValueError, '不属于'):
                self.ui.subagent_log(child_id)
        self.parent.append('subagent/end', child_session='unsaved', session_path=None)
        with self.assertRaisesRegex(ValueError, '未保存'):
            self.ui.subagent_log('unsaved')
        path = Path(self.result.session_path)
        original = path.read_bytes()
        path.write_bytes(self.parent.source_path.read_bytes())
        with self.assertRaisesRegex(ValueError, '不匹配'):
            self.ui.subagent_log(self.result.session_id)
        path.write_bytes(original)
        path.unlink()
        with self.assertRaises(ValueError):
            self.ui.subagent_log(self.result.session_id)

    async def test_http_view_log_does_not_switch_conversation(self):
        self.ui.port = 0
        self.ui.start()
        base = f'http://127.0.0.1:{self.ui.port}'
        status, data = await asyncio.to_thread(
            post_json, base, '/api/subagent/log',
            {'session': self.parent.id, 'child_id': self.result.session_id})
        self.assertEqual(status, 200)
        self.assertIn('5556', data['content'])
        self.assertIs(self.ui.session, self.parent)
        other = self.ctx.sessions.create()
        self.ui.attach(other)
        status, _ = await asyncio.to_thread(
            post_json, base, '/api/subagent/log',
            {'session': other.id, 'child_id': self.result.session_id})
        self.assertEqual(status, 400)
