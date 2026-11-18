import importlib.util,sys,json,hashlib
from pathlib import Path
from unittest.mock import Mock,patch
import pytest

SOURCE=Path(__file__).resolve().parents[1]/'app/providers/qwen_server_provider.py'
spec=importlib.util.spec_from_file_location('desktop_native_provider_test',SOURCE)
provider_module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=provider_module;spec.loader.exec_module(provider_module)

@pytest.fixture
def profile(tmp_path,monkeypatch):
    model=tmp_path/'model.gguf';model.write_bytes(b'GGUFcontrolled test fixture; not neural weights')
    binary=tmp_path/'llama-server.exe';binary.write_bytes(b'controlled fixture; never executed')
    key=tmp_path/'key';key.write_text('a'*48)
    data={'schema_version':'niblit-local-model/1','version':'fixture-v1','architecture':'qwen2','tokenizer':'embedded-gguf','model_id':'qwen2.5-coder-3b',
      'model':{'path':str(model),'sha256':hashlib.sha256(model.read_bytes()).hexdigest()},
      'executable':{'path':str(binary),'sha256':hashlib.sha256(binary.read_bytes()).hexdigest()},
      'runtime':{'context_length':4096,'threads':1,'gpu_layers':0}}
    path=tmp_path/'active.json';path.write_text(json.dumps(data))
    monkeypatch.setenv('NIBLIT_ACTIVE_MODEL_PROFILE',str(path));monkeypatch.setenv('NIBLIT_PORTAL_GGUF_KEY_FILE',str(key))
    monkeypatch.setenv('NIBLIT_SERVICE_LOG_DIR',str(tmp_path/'logs'));monkeypatch.delenv('NIBLIT_FALLBACK_MODEL_PROFILE',raising=False)
    return provider_module.QwenServerProvider(model_path=str(model),backend_bin=str(binary),port=8866),data,path

@pytest.mark.parametrize('change',[{'architecture':'unverified-native'},{'tokenizer':'other'},{'model_id':'unverified'},
 {'version':''},{'runtime':{'context_length':512,'threads':1,'gpu_layers':0}},
 {'runtime':{'context_length':4096,'threads':1,'gpu_layers':1}}])
def test_invalid_profile_never_executes(profile,change):
    provider,data,path=profile;data.update(change);path.write_text(json.dumps(data))
    with patch.object(provider_module.subprocess,'Popen') as send:
        assert provider._start_native_windows() is False
        send.assert_not_called()

def test_tampered_weights_denied(profile):
    provider,data,path=profile;data['model']['sha256']='0'*64;path.write_text(json.dumps(data))
    with patch.object(provider_module.subprocess,'Popen') as send:
        assert provider._start_native_windows() is False
        send.assert_not_called()

def test_compatible_profile_and_invalid_experiment_fallback(profile,monkeypatch):
    provider,data,path=profile
    fallback=path.with_name('known-good.json');fallback.write_text(json.dumps(data))
    path.write_text(json.dumps(dict(data,architecture='unverified-native')))
    monkeypatch.setenv('NIBLIT_FALLBACK_MODEL_PROFILE',str(fallback))
    process=Mock();process.poll.return_value=None
    response=Mock(status_code=200);response.json.return_value={'data':[{'id':'qwen2.5-coder-3b'}]}
    with patch.object(provider_module.subprocess,'Popen',return_value=process) as send,patch('socket.socket') as socket,patch('httpx.Client') as client:
        socket.return_value.__enter__.return_value.connect_ex.return_value=1
        client.return_value.__enter__.return_value.get.return_value=response
        assert provider._start_native_windows() is True
        command=send.call_args.args[0]
        assert command[0]==provider.backend_bin and '--api-key-file' in command and 'bash' not in command
        assert provider.profile_version=='fixture-v1'
        provider.stop();process.terminate.assert_called_once()
