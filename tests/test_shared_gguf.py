import json,httpx,pytest
from app.shared_gguf import SharedGGUFEngine,delegated_token
from fastapi import HTTPException

def setup(monkeypatch,tmp_path,kind='ok'):
    key=tmp_path/'key';key.write_text('synthetic-test-key-at-least-32-chars')
    monkeypatch.setenv('NIBLIT_CLOUD_GGUF_KEY_FILE',str(key));calls=[];real=httpx.Client
    def handler(req):
        body=json.loads(req.content);calls.append((req.url.port,req.url.path,body))
        if req.url.port==8766:
            if kind=='reject':return httpx.Response(400,json={'error':'controlled context rejection'})
            if kind=='timeout':raise httpx.ReadTimeout('synthetic',request=req)
            return httpx.Response(200,json={'choices':[{'message':{'content':'4','tool_calls':[]},'finish_reason':'stop'}],'usage':{'total_tokens':4}})
        if req.url.path.endswith('acquire'):return httpx.Response(200,json={'token':'s'*40})
        return httpx.Response(400 if kind=='denied' else 200,json={'state':'ok'})
    monkeypatch.setattr('app.shared_gguf.httpx.Client',lambda **kw:real(transport=httpx.MockTransport(handler),**kw))
    return calls

def test_delegation_preserves_parent(monkeypatch,tmp_path):
    calls=setup(monkeypatch,tmp_path);marker=delegated_token.set('p'*40)
    try:r=SharedGGUFEngine().chat([],0,8)
    finally:delegated_token.reset(marker)
    assert r.text=='4' and len(calls)==3 and calls[0][2]['token']=='p'*40 and calls[-1][2]['complete']
    assert not any(c[1].endswith(('acquire','finish')) for c in calls)

def test_direct_client_owns_one_task(monkeypatch,tmp_path):
    calls=setup(monkeypatch,tmp_path);assert SharedGGUFEngine().chat([],0,8).text=='4'
    assert calls[0][1].endswith('acquire') and calls[-1][1].endswith('finish')

@pytest.mark.parametrize('kind',['denied','timeout'])
def test_failure_no_retry_or_fallback(monkeypatch,tmp_path,kind):
    calls=setup(monkeypatch,tmp_path,kind);marker=delegated_token.set('p'*40)
    try:
        with pytest.raises((HTTPException,httpx.ReadTimeout)):SharedGGUFEngine().chat([],0,8)
    finally:delegated_token.reset(marker)
    assert sum(c[0]==8766 for c in calls)==(1 if kind=='timeout' else 0)
    if kind=='timeout':assert calls[-1][2]['uncertain']


@pytest.mark.parametrize('stream',[False,True])
def test_real_cloud_route_propagates_lease_including_stream(monkeypatch,tmp_path,stream):
    from fastapi.testclient import TestClient
    from app.main import ModelManager,create_app
    calls=setup(monkeypatch,tmp_path)
    monkeypatch.setenv('NIBLIT_CLOUD_SHARED_GGUF','1')
    manager=ModelManager({'qwen2.5-coder-3b':'not-loaded.gguf'},'qwen2.5-coder-3b')
    with TestClient(create_app(model_manager=manager)) as client:
        r=client.post('/v1/chat/completions',headers={'x-niblit-task-token':'p'*40},json={'model':'qwen2.5-coder-3b','messages':[{'role':'user','content':'2+2'}],'max_tokens':8,'stream':stream})
    assert r.status_code==200
    assert calls[0][2]['token']=='p'*40
    assert not any(c[1].endswith(('acquire','finish')) for c in calls)
    assert '[DONE]' in r.text if stream else r.json()['choices'][0]['message']['content']=='4'


def test_definite_model_rejection_completes_failed_child_without_quarantine(monkeypatch,tmp_path):
    calls=setup(monkeypatch,tmp_path,'reject');marker=delegated_token.set('p'*40)
    try:
        with pytest.raises(HTTPException) as error:SharedGGUFEngine().chat([],0,8)
    finally:delegated_token.reset(marker)
    assert error.value.status_code==400
    assert calls[-1][2]['complete'] and calls[-1][2]['uncertain'] is False
    assert sum(c[0]==8766 for c in calls)==1

def test_tool_schemas_consume_context_without_being_removed():
    from app.main import ModelManager
    manager=ModelManager({'qwen2.5-coder-3b':'unused'},'qwen2.5-coder-3b')
    tools=[{'type':'function','function':{'name':'read_status','description':'description '*100,'parameters':{'type':'object'}}}]
    messages=[{'role':'user','content':'context '*1000}]
    plain=manager.estimate_inference(messages,512)
    with_tools=manager.estimate_inference(messages,512,tools=tools)
    assert with_tools['prompt_tokens_estimate']>plain['prompt_tokens_estimate']
    assert tools[0]['function']['name']=='read_status'
    with pytest.raises(HTTPException) as error:manager.estimate_inference([],512,tools=[{'description':'x'*1000000}])
    assert error.value.status_code==413


def test_shared_and_embedded_engines_share_existing_text_tool_normalization():
    from app.main import normalize_tool_calls
    tools=[{'type':'function','function':{'name':'runtime_status'}}]
    for content in ('{"name":"runtime_status","arguments":{}}','```json\n{"name":"runtime_status","args":{}}\n```'):
        calls=normalize_tool_calls(content,None,tools)
        assert calls[0]['function']['name']=='runtime_status'
        assert json.loads(calls[0]['function']['arguments'])=={}
    assert normalize_tool_calls('{"name":"unregistered","arguments":{}}',None,tools) is None
    assert normalize_tool_calls('{"name":"runtime_status","arguments":{}}',None,None) is None
