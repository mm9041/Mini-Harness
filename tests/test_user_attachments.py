import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from mini_harness.history import ConversationHistory
from mini_harness.session import Session, SessionsService
from mini_harness.user_messages import ATTACHMENT_NOTICE, user_message_view
from mini_harness.webui import ui_message


class UserAttachmentTests(unittest.TestCase):
    def setUp(self):
        self.files = [{'name': '新建 文本文档.txt', 'size': 6606,
                       'path': 'E:/mh-test/.mini-harness/uploads/' + 'a' * 32 + '/新建 文本文档.txt'}]

    def test_restart_preserves_clean_text_and_model_file_context(self):
        with tempfile.TemporaryDirectory() as root:
            sessions = SessionsService(root)
            session = sessions.create()
            event = session.append('user/message', text='第一行内容是什么', attachments=self.files)
            sessions.save(session)
            reopened = SessionsService(root).open(session.source_path)
            event = reopened.events_of('user/message')[0]
            self.assertEqual(event.data['text'], '第一行内容是什么')
            view = ui_message(event)
            self.assertEqual(view['text'], '第一行内容是什么')
            self.assertEqual(view['attachments'], [{'name': self.files[0]['name'], 'size': 6606}])
            self.assertNotIn('path', str(view))
            model = reopened.derive_messages()[0].content
            self.assertIn('第一行内容是什么', model)
            self.assertIn(self.files[0]['path'], model)
            self.assertIn('请使用文件工具读取', model)
            self.assertEqual(ConversationHistory(Path(root)).list()[0]['title'], '第一行内容是什么')

    def test_old_uploads_hide_only_generated_suffix_without_changing_model_text(self):
        text = '第一行内容是什么' + ATTACHMENT_NOTICE + json.dumps(self.files, ensure_ascii=False)
        session = Session('legacy')
        event = session.append('user/message', text=text)
        self.assertEqual(ui_message(event)['text'], '第一行内容是什么')
        self.assertEqual(len(ui_message(event)['attachments']), 1)
        self.assertEqual(session.derive_messages()[0].content, text)
        self.assertEqual(event.data['text'], text)

    def test_similar_prose_or_invalid_metadata_is_not_removed(self):
        for suffix in ('not json', '{}', '[]', '[null]',
                       json.dumps([{'name': 'file', 'size': 2, 'path': '/tmp/file'}])):
            text = '解释这段提示' + ATTACHMENT_NOTICE + suffix
            self.assertEqual(user_message_view({'text': text}), {'text': text, 'attachments': []})
        # Explicit metadata wins even if the user text quotes a legacy upload.
        text = 'quoted' + ATTACHMENT_NOTICE + json.dumps(self.files)
        self.assertEqual(user_message_view({'text': text, 'attachments': []})['text'], text)

    def test_attachment_only_message_keeps_model_instruction_and_filename_title(self):
        with tempfile.TemporaryDirectory() as root:
            sessions = SessionsService(root)
            session = sessions.create()
            session.append('user/message', text='', attachments=self.files)
            sessions.save(session)
            self.assertTrue(session.derive_messages()[0].content.startswith('请查看上传的文件。'))
            self.assertEqual(ConversationHistory(Path(root)).list()[0]['title'], self.files[0]['name'])

    @unittest.skipUnless(shutil.which('node'), 'Node.js required')
    def test_frontend_renders_text_and_files_separately(self):
        source = (Path(__file__).resolve().parents[1] / 'mini_harness/webui_static/index.html').read_text(encoding='utf-8')
        handler = source[source.index('function addUser('):source.index('function ensureAssistant(')]
        script = r"""
const assert=require('node:assert/strict');
const el=(tag,cls,text)=>({tag,cls,text,children:[],appendChild(n){this.children.push(n);},append(...ns){this.children.push(...ns);}});
const transcript=el('main');transcript.querySelector=()=>null;
const title={textContent:'新对话'}, $=()=>title, t=s=>s, scroll=()=>{};
""" + handler + r"""
addUser('第一行内容是什么',[{name:'<script>.txt',size:6606},{name:'second.txt',size:0}]);
assert.equal(transcript.children[0].children[0].text,'第一行内容是什么');
const cards=transcript.children[0].children[1].children;
assert.equal(cards.length,2);assert.equal(cards[0].children[0].text,'<script>.txt');
addUser('',[{name:'only.txt',size:1}]);
assert.equal(transcript.children[1].children.length,1);
assert.equal(transcript.children[1].children[0].cls,'user-files');
addUser('plain');assert.equal(transcript.children[2].children.length,1);
"""
        result = subprocess.run(['node'], input=script, capture_output=True, encoding='utf-8', timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
