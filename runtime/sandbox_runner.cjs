// Trusted bridge, installed outside the writable workspace alongside the DSH runtime.
// Set the hidden console's code page before the restricted PowerShell child starts;
// read-only PowerShell can use ConstrainedLanguage and cannot set it via .NET itself.
const { pathToFileURL } = require('node:url');
try {
  const kernel32 = require('koffi').load('kernel32.dll');
  const setOutput = kernel32.func('bool __stdcall SetConsoleOutputCP(uint32_t codePage)');
  const setInput = kernel32.func('bool __stdcall SetConsoleCP(uint32_t codePage)');
  if (!setOutput(65001) || !setInput(65001)) throw new Error('console UTF-8 setup failed');
  import(pathToFileURL(require.resolve('@deepseek-ai/dsh-sandbox-windows-acl/runner')).href)
    .catch(error => { console.error('windows-acl-run:', error.message); process.exitCode = 127; });
} catch (error) {
  console.error('windows-acl-run:', error.message);
  process.exitCode = 127;
}
