import asyncio
import importlib.util
import sys
import base64
import io
import json
import os
import shutil
import unittest
from unittest.mock import patch

from mini_harness.adapters.mock import ScriptedAdapter
from mini_harness.llm import ToolCall
from .support import adapter_plugin_for, build_context_with, make_temp_dir, remove_tree


class _ToolFixture(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.cwd = make_temp_dir()
        self.ctx = build_context_with(adapter_plugin_for(ScriptedAdapter([])), self.cwd,
                                      approval='allow', session_root=self.cwd/'sessions', compaction=False)
        self.session = self.ctx.sessions.create('owner')

    async def asyncTearDown(self):
        self.ctx.dispose()
        await remove_tree(self.cwd)

    async def call(self, name, **args):
        return await self.ctx.tools.execute(ToolCall('call', name, args), session=self.session, cwd=self.cwd)


class CatalogToolsTests(_ToolFixture):
    async def test_read_is_line_numbered_and_paginated(self):
        (self.cwd/'text.txt').write_text('one\ntwo\nthree\n', encoding='utf-8')
        result = await self.call('read', file_path='text.txt', offset=2, limit=1)
        self.assertFalse(result.is_error)
        self.assertIn('2\t' + 'two', result.content)
        self.assertNotIn('one', result.content)
        self.assertNotIn('three', result.content)
        self.assertTrue((await self.call('read', file_path='text.txt', offset=0)).is_error)

    async def test_write_and_literal_edit_preserve_crlf_and_bom(self):
        original = '\ufefffirst\r\nsecond\r\n'
        result = await self.call('write', file_path='a.txt', content=original)
        self.assertFalse(result.is_error)
        result = await self.call('edit', file_path='a.txt', old_string='second', new_string='changed')
        self.assertFalse(result.is_error)
        self.assertEqual((self.cwd/'a.txt').read_bytes(), original.replace('second','changed').encode())

    async def test_ambiguous_edit_requires_replace_all(self):
        path = self.cwd/'a.txt'
        path.write_text('same same', encoding='utf-8')
        result = await self.call('edit', file_path='a.txt', old_string='same', new_string='new')
        self.assertTrue(result.is_error)
        self.assertEqual(path.read_text(), 'same same')
        result = await self.call('edit', file_path='a.txt', old_string='same', new_string='new', replace_all=True)
        self.assertFalse(result.is_error)
        self.assertEqual(path.read_text(), 'new new')

    async def test_failed_atomic_replace_keeps_original(self):
        path = self.cwd/'a.txt'
        path.write_text('original', encoding='utf-8')
        with patch('mini_harness.builtin_tools.files.os.replace', side_effect=OSError('disk error')):
            result = await self.call('write', file_path='a.txt', content='changed')
        self.assertTrue(result.is_error)
        self.assertEqual(path.read_text(), 'original')
        self.assertEqual(list(self.cwd.glob('.harness-edit-*')), [])

    async def test_search_includes_hidden_and_ignored_files(self):
        (self.cwd/'.gitignore').write_text('ignored/\n', encoding='utf-8')
        for folder in ['.hidden', 'ignored']:
            path = self.cwd/folder
            path.mkdir()
            (path/'code.py').write_text('x = 1\nTARGET = 2\n', encoding='utf-8')
        result = await self.call('glob', pattern='**/*.py')
        self.assertIn('.hidden', result.content)
        self.assertIn('ignored', result.content)
        result = await self.call('grep', pattern='^target', glob='**/*.py', ignore_case=True)
        self.assertIn('code.py:2:TARGET', result.content)
        self.assertEqual(result.content.count('TARGET'), 2)
        self.assertTrue((await self.call('glob', pattern='../*')).is_error)
        self.assertTrue((await self.call('grep', pattern='[')).is_error)

    async def test_mutating_tools_share_approval(self):
        self.ctx.approval.set_mode('deny')
        for name, args in [
            ('write', {'file_path':'denied.txt','content':'x'}),
            ('edit', {'file_path':'denied.txt','old_string':'x','new_string':'y'}),
            ('pwsh', {'command':'Write-Output should-not-run'}),
        ]:
            self.assertTrue((await self.call(name, **args)).is_error)
        self.assertFalse((self.cwd/'denied.txt').exists())
        self.assertEqual(self.ctx.jobs.list(self.session.id), [])

    async def test_workspace_escape_is_rejected(self):
        for name, args in [('read',{'file_path':'../outside.txt'}),('write',{'file_path':'../outside.txt','content':'bad'}),
                           ('edit',{'file_path':'../outside.txt','old_string':'a','new_string':'b'}),('grep',{'path':'..','pattern':'x'}),
                           ('read_image',{'file_path':'../outside.png'}),('pwsh',{'command':'Write-Output x','cwd':'..'})]:
            with self.subTest(tool=name):
                self.assertTrue((await self.call(name, **args)).is_error)

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow optional dependency not installed")
    async def test_images_all_formats_resize_and_reach_model_after_tool_batch(self):
        from PIL import Image
        for format, extension in [('PNG','png'),('JPEG','jpg'),('WEBP','webp'),('GIF','gif')]:
            Image.new('RGB',(2200,1100),'red').save(self.cwd/f'image.{extension}', format=format)
            result = await self.call('read_image', file_path=f'image.{extension}', max_edge=800)
            self.assertFalse(result.is_error, result.content)
            self.assertEqual((result.images[0]['width'],result.images[0]['height']), (800,400))
            raw = base64.b64decode(result.images[0]['data_url'].split(',',1)[1])
            with Image.open(io.BytesIO(raw)) as image:
                self.assertEqual(image.size,(800,400))
        adapter = ScriptedAdapter([
            {'tool_calls':[{'id':'image','name':'read_image','arguments':{'file_path':'image.png'}},
                           {'id':'text','name':'read','arguments':{'file_path':'note.txt'}}]},
            {'text':'read both'},
        ])
        (self.cwd/'note.txt').write_text('text',encoding='utf-8')
        self.ctx.llm.register_adapter('images',adapter); self.ctx.llm.use('images')
        result = await self.ctx.agents.create(self.session).run('read image and text')
        self.assertEqual(result.stopped,'final')
        messages = adapter.requests[-1].messages
        self.assertEqual([m.role for m in messages],['user','assistant','tool','tool','user'])
        self.assertEqual(messages[-1].to_wire()['content'][1]['type'],'image_url')
        path = self.ctx.sessions.save(self.session)
        self.assertEqual(self.ctx.sessions.open(path).derive_messages()[-2].images,messages[-1].images)


@unittest.skipUnless(shutil.which('pwsh') or shutil.which('powershell'), 'PowerShell not installed')
class PowerShellJobsTests(_ToolFixture):
    async def test_explicit_null_timeouts_use_defaults(self):
        result = await self.call('pwsh', command="Write-Output 'null timeout'", timeout=None, background=True)
        self.assertFalse(result.is_error, result.content)
        job_id = json.loads(result.content)['job_id']
        result = await self.call('job_output', job_id=job_id, wait=True, timeout=None)
        self.assertFalse(result.is_error, result.content)
        self.assertEqual(json.loads(result.content)['status'], 'completed')
        self.assertIn('null timeout', json.loads(result.content)['output'])

    @unittest.skipUnless(os.name == 'nt' and shutil.which('powershell'), 'Windows PowerShell fallback required')
    async def test_windows_powershell_fallback_suppresses_progress_xml(self):
        executable = shutil.which('powershell')
        with patch('mini_harness.jobs.shutil.which', side_effect=lambda name: executable if name == 'powershell' else None):
            result = await self.call('pwsh', command="Write-Progress -Activity 'test progress' -Status 'working'; Write-Output 'hello'", timeout=20)
        self.assertFalse(result.is_error, result.content)
        output = json.loads(result.content)['output']
        self.assertEqual(output.strip(), 'hello')
        self.assertNotIn('CLIXML', output)

    async def test_foreground_unicode_and_nonzero_exit(self):
        result = await self.call('pwsh',command="Write-Output '你好 PowerShell'",timeout=10)
        self.assertFalse(result.is_error, result.content)
        self.assertIn('你好',json.loads(result.content)['output'])
        result = await self.call('pwsh',command='exit 7',timeout=10)
        self.assertTrue(result.is_error)
        self.assertEqual(json.loads(result.content)['returncode'],7)

    async def test_background_read_wait_and_session_ownership(self):
        result = await self.call('pwsh',command="Write-Output 'started'; Start-Sleep -Seconds 2; Write-Output 'finished'",background=True,timeout=10)
        self.assertFalse(result.is_error,result.content)
        job_id = json.loads(result.content)['job_id']
        self.assertEqual(len(json.loads((await self.call('job_list')).content)),1)
        result = await self.call('job_output',job_id=job_id,wait=True,timeout=.1)
        self.assertEqual(json.loads(result.content)['status'],'running')
        with self.assertRaises(ValueError):
            self.ctx.jobs.get(job_id,'another-session')
        result = await self.call('job_output',job_id=job_id,wait=True,timeout=10)
        output = json.loads(result.content)
        self.assertEqual(output['status'],'completed')
        self.assertIn('finished',output['output'])
        self.assertEqual(json.loads((await self.call('job_output',job_id=job_id,offset=output['next_offset'])).content)['output'],'')

    async def test_timeout_and_explicit_kill_stop_process(self):
        result = await self.call('pwsh',command='Start-Sleep -Seconds 20',timeout=.5)
        self.assertTrue(result.is_error)
        self.assertEqual(json.loads(result.content)['status'],'timed_out')
        result = await self.call('pwsh',command='Start-Sleep -Seconds 20',background=True,timeout=30)
        job_id = json.loads(result.content)['job_id']
        result = await self.call('job_kill',job_id=job_id)
        self.assertEqual(json.loads(result.content)['status'],'killed')
        self.assertIsNotNone(self.ctx.jobs.get(job_id,self.session.id).process.poll())

    async def test_cancellation_and_service_shutdown(self):
        task = asyncio.create_task(self.call('pwsh',command='Start-Sleep -Seconds 20',timeout=30))
        while not self.ctx.jobs.list(self.session.id):
            await asyncio.sleep(.01)
        self.ctx.interrupt.request('test cancellation')
        result = await asyncio.wait_for(task,5)
        self.assertTrue(result.is_error)
        self.ctx.interrupt.reset()
        result = await self.call('pwsh',command='Start-Sleep -Seconds 20',background=True,timeout=30)
        job = self.ctx.jobs.get(json.loads(result.content)['job_id'],self.session.id)
        await asyncio.to_thread(self.ctx.jobs.close)
        self.assertTrue(job.done.is_set())
        self.assertIsNotNone(job.process.poll())

    async def test_job_output_does_not_split_utf8_characters(self):
        result = await self.call('pwsh', command="Write-Output '你好'", timeout=10)
        job = self.ctx.jobs.get(json.loads(result.content)['job_id'], self.session.id)
        first = self.ctx.jobs.output(job, limit=1)
        self.assertEqual(first['output'], '你')
        self.assertEqual(first['next_offset'], 3)
        second = self.ctx.jobs.output(job, offset=3, limit=1)
        self.assertEqual(second['output'], '好')

    @unittest.skipUnless(os.name == 'nt', 'Windows job-object process-tree test')
    async def test_kill_also_terminates_spawned_child(self):
        import ctypes
        from ctypes import wintypes
        executable = sys.executable.replace("'", "''")
        command = f"$child = Start-Process -FilePath '{executable}' -ArgumentList '-c \"import time; time.sleep(30)\"' -WindowStyle Hidden -PassThru; Write-Output $child.Id; Start-Sleep -Seconds 30"
        result = await self.call('pwsh', command=command, background=True, timeout=40)
        job = self.ctx.jobs.get(json.loads(result.content)['job_id'], self.session.id)
        child_id = None
        for _ in range(100):
            output = self.ctx.jobs.output(job)['output'].strip()
            if output.isdigit():
                child_id = int(output)
                break
            await asyncio.sleep(.05)
        self.assertIsNotNone(child_id, self.ctx.jobs.output(job))
        dll = ctypes.WinDLL('kernel32', use_last_error=True)
        dll.OpenProcess.argtypes = [wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        dll.OpenProcess.restype = wintypes.HANDLE
        dll.WaitForSingleObject.argtypes = [wintypes.HANDLE,wintypes.DWORD]
        dll.WaitForSingleObject.restype = wintypes.DWORD
        dll.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = dll.OpenProcess(0x100000, False, child_id)
        self.assertTrue(handle)
        try:
            self.assertEqual(dll.WaitForSingleObject(handle, 0), 258)
            await self.call('job_kill', job_id=job.id)
            self.assertEqual(dll.WaitForSingleObject(handle, 3000), 0)
        finally:
            dll.CloseHandle(handle)
