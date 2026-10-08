"""Exercise the shipped frontend handlers with deterministic HTTP/SSE ordering."""
from pathlib import Path
import shutil
import subprocess
import unittest


HTML = Path(__file__).resolve().parents[1] / 'mini_harness/webui_static/index.html'


@unittest.skipUnless(shutil.which('node'), 'Node.js required for frontend state regression tests')
class FrontendBusyTests(unittest.TestCase):
    def run_scenario(self, scenario):
        source = HTML.read_text(encoding='utf-8')
        def section(start, end):
            return source[source.index(start):source.index(end, source.index(start))]
        handlers = '\n'.join([
            section('function setBusy(value)', '/* ---------------------------------------------------------------- 事件分发 */'),
            section('function handle(message)', 'function showQuestion('),
            section('async function refreshStatus()', 'async function loadModels()'),
            section("$('send').onclick =", "$('openWorkspace').onclick ="),
        ])
        setup = r"""
const assert = require('node:assert/strict');
const nodes = new Map();
function $(id) {
  if (!nodes.has(id)) nodes.set(id, {value:'', disabled:false, hidden:false, title:'', innerHTML:'',
    classList:{toggle(){}}, attributes:{}, setAttribute(k,v){this.attributes[k]=v;}, remove(){}});
  return nodes.get(id);
}
const document = {getElementById:id=>nodes.get(id),querySelectorAll:()=>[]};
let busy=false, sessionId='current', stateRevision=0, statusRequestId=0, choosingDirectory=false,
    connected=true, sendingAttachments=false, attachments=[], followLatest=false;
const t=s=>s, updateClock=()=>{}, renderAttachments=()=>{}, saveDraft=()=>{}, resizeInput=()=>{};
const encodeFile=async file=>file, addNotice=()=>{};
let applied=[];
const applyStatus=s=>applied.push(s);
let fetch, post;
function deferred(){let resolve,reject;const promise=new Promise((r,j)=>{resolve=r;reject=j});return {promise,resolve,reject};}
function response(busy){return {ok:true,json:async()=>({busy})};}
"""
        script = setup + '\n' + handlers + '\nconst watchdog=setTimeout(()=>process.exit(2),2000);\n(async()=>{' + scenario + r"""
})().then(()=>clearTimeout(watchdog)).catch(error=>{clearTimeout(watchdog);console.error(error.stack);process.exitCode=1;});
"""
        result = subprocess.run(['node', '-e', script], capture_output=True, text=True,
                                encoding='utf-8', timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_late_busy_snapshot_cannot_overwrite_completion_push(self):
        self.run_scenario("""
busy=true;
const pending=deferred(); fetch=()=>pending.promise;
const read=refreshStatus();
handle({kind:'busy',value:false});
pending.resolve(response(true)); await read;
assert.equal(busy,false); assert.equal($('send').disabled,false);
assert.equal(applied.length,0);
""")

    def test_running_conversation_keeps_new_chat_controls_enabled(self):
        self.run_scenario("""
setBusy(true);
assert.equal($('newSession').disabled,false);
assert.equal($('createChat').disabled,false);
assert.equal($('model').disabled,true);
assert.equal($('reasoning').disabled,true);
""")

    def test_late_events_from_other_conversations_are_ignored(self):
        self.run_scenario("""
handle({kind:'busy',value:true,session:'previous'});
handle({kind:'delta',text:'must not render',session:'previous'});
assert.equal(busy,false); assert.equal(stateRevision,0);
handle({kind:'busy',value:true,session:'current'});
assert.equal(busy,true);
""")

    def test_fast_command_releases_button_before_status_request_finishes(self):
        self.run_scenario("""
$('input').value='/tools';
const pending=deferred(), started=deferred();
fetch=()=>{started.resolve();return pending.promise;};
post=async path=>{assert.equal(path,'/api/message');handle({kind:'busy',value:true});handle({kind:'busy',value:false});return {ok:true};};
const send=$('send').onclick(); await started.promise;
assert.equal(sendingAttachments,false); assert.equal(busy,false);
assert.equal($('send').disabled,false);
pending.resolve(response(false)); await send;
""")

    def test_stop_reconciles_when_server_reports_already_idle(self):
        self.run_scenario("""
setBusy(true);
post=async path=>{assert.equal(path,'/api/interrupt');return {ok:false};};
fetch=async()=>response(false);
await $('send').onclick();
assert.equal(busy,false); assert.equal($('send').disabled,false);
""")

    def test_newer_http_request_supersedes_an_older_snapshot(self):
        self.run_scenario("""
const older=deferred(), newer=deferred();let calls=0;
fetch=()=>++calls===1?older.promise:newer.promise;
const first=refreshStatus(),second=refreshStatus();
newer.resolve(response(false));await second;
older.resolve(response(true));await first;
assert.equal(busy,false);assert.equal(applied.length,1);
""")

    def test_new_submission_invalidates_old_idle_read_without_waiting_for_sse(self):
        self.run_scenario("""
const oldRead=deferred(), postPending=deferred(), postStarted=deferred();let reads=0;
fetch=()=>++reads===1?oldRead.promise:Promise.resolve(response(true));
const previous=refreshStatus();$('input').value='new task';
post=()=>{postStarted.resolve();return postPending.promise;};
const send=$('send').onclick();await postStarted.promise;
oldRead.resolve(response(false));await previous;
assert.equal(busy,true);
postPending.resolve({ok:true});await send;
assert.equal(busy,true);
""")


if __name__ == '__main__':
    unittest.main()
