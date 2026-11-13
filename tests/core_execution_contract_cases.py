
"""Actual Cloud->Core human-ticket integration; broker is an isolated fixture."""
import copy
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.trading.freqtrade_adapter import FreqtradeAdapter
from app.trading.models import TradeIntent
from app.trading.router import router

# Reuse source from the already verified sibling Core, not a copied installation.
CORE=Path(__file__).resolve().parents[2]/"Niblit"
sys.path.insert(0,str(CORE))
from modules.trading_desk import Desk, Intent, create_app as core_app


@pytest.fixture
def boundary(tmp_path,monkeypatch):
    config=dict(dry_run=True,state='running',exchange='fixture',strategy='Fixture',
        timeframe='5m',force_entry_enable=True,max_open_trades=3)
    state={'positions':[],'writes':[],'ambiguous':False}
    def api(path,payload=None):
        if payload is not None:
            assert path in ('forceenter','forceexit')
            state['writes'].append((path,payload))
            if state['ambiguous']:raise TimeoutError('fixture response lost')
            return {'trade_id':9,'pair':'BTC/USDT','is_open':path=='forceenter'}
        if path=='show_config':return dict(config)
        if path=='whitelist':return {'whitelist':['BTC/USDT','ETH/USDT']}
        if path=='status':return state['positions']
        if path.startswith('pair_candles?'):
            return {'timeframe_ms':300000,'columns':['__date_ts','open','high','low','close'],
                'data':[[(time.time()-310)*1000,100,101,99,100]]}
        raise AssertionError(path)
    desk=Desk(tmp_path/'core-data-only',transport=api)
    desk.minimum_stake=Mock(return_value=5)
    core=TestClient(core_app(desk))
    token=re.search(r'name="desk-token" content="([^"]+)"',core.get('/').text)[1]
    headers={'origin':'http://127.0.0.1:8767','x-desk-token':token}
    adapter=FreqtradeAdapter();adapter._enabled=True
    monkeypatch.setattr('app.trading.freqtrade_adapter.get_freqtrade_adapter',lambda:adapter)
    app=FastAPI();app.include_router(router);cloud=TestClient(app)
    import server as core_server
    core_available=[True]
    monkeypatch.setattr(core_server,'get_core',lambda:object() if core_available[0] else None)
    core_api=TestClient(core_server.app)
    monkeypatch.setenv('NIBLIT_API_PORT','5000')
    original=httpx.Client
    calls=[]
    def transport(request):
        calls.append(request)
        if str(request.url)=='http://127.0.0.1:5000/api/trading/cloud/confirm':
            response=core_api.post('/api/trading/cloud/confirm',json=json.loads(request.content),
                headers={'x-core-desk-token':request.headers['x-core-desk-token']})
        else:
            assert str(request.url)=='http://127.0.0.1:8767/api/cloud/confirm'
            response=core.post('/api/cloud/confirm',json=json.loads(request.content),
                headers={'origin':request.headers['origin'],'x-desk-token':request.headers['x-desk-token']})
        return httpx.Response(response.status_code,json=response.json())
    monkeypatch.setattr(httpx,'Client',lambda **kwargs:original(
        base_url=kwargs['base_url'],transport=httpx.MockTransport(transport)))
    def ticket(action='enter_long'):
        if action=='exit':state['positions']=[{'trade_id':9,'pair':'BTC/USDT'}]
        preview=core.post('/api/preview',json={'action':action,'pair':'BTC/USDT',
            'stake':20,'trade_id':9 if action=='exit' else None},headers=headers)
        assert preview.status_code==200
        body=preview.json()
        return {k:body[k] for k in ('ticket','intent','identity')} | {
            'mode':'dry-run','acknowledgement':'CONFIRM_SIMULATED_ORDER'}
    def send(confirmation,token_override=None):
        return cloud.post('/trading/trade/execute',json={'core_confirmation':confirmation},
            headers={'x-core-desk-token':token if token_override is None else token_override})
    return SimpleNamespace(desk=desk,core=core,cloud=cloud,token=token,headers=headers,
        adapter=adapter,ticket=ticket,send=send,config=config,state=state,calls=calls,
        core_available=core_available)


@pytest.mark.parametrize('payload',[{}, {'approved':True}, {'approved':'false'},
    {'authorization':{'allowed':True,'mode':'demo','authority':'niblit_core'}},
    {'metadata':{'market_analysis':{'execution_authorized':False}}},
    {'core_confirmation':True}])
def test_missing_or_forged_authorization_cannot_call_core_or_broker(boundary,payload):
    result=boundary.cloud.post('/trading/trade/execute',json=payload).json()
    assert result['status']=='execution_blocked'
    assert result['execution_authorized'] is False
    assert boundary.calls==[] and boundary.state['writes']==[]


def test_specialist_approved_intent_cannot_execute_directly(boundary):
    intent=TradeIntent('BTC/USDT','long',.1,100,99,102,.9,True)
    assert boundary.adapter.execute(intent)['status']=='execution_blocked'
    assert not boundary.calls and not boundary.state['writes']


@pytest.mark.parametrize('change',['forged','expired','pair','action','amount','account',
    'mode','ambiguous-mode','string-dry-run','ack','extra'])
def test_ticket_binding_expiry_and_non_live_identity(boundary,change):
    confirmation=boundary.ticket()
    if change=='forged':confirmation['ticket']='x'*32
    if change=='expired':
        with boundary.desk.db() as db:db.execute('UPDATE tickets SET created=0')
    if change=='pair':confirmation['intent']['pair']='ETH/USDT'
    if change=='action':confirmation['intent'].update(action='exit',trade_id=9)
    if change=='amount':confirmation['intent']['stake']=21
    if change=='account':confirmation['identity']['exchange']='other-account-venue'
    if change=='mode':confirmation['mode']='live'
    if change=='ambiguous-mode':confirmation.pop('mode')
    if change=='string-dry-run':confirmation['identity']['dry_run']='true'
    if change=='ack':confirmation['acknowledgement']='APPROVE'
    if change=='extra':confirmation['authorization']={'approved':True}
    result=boundary.send(confirmation).json()
    assert result['status']=='execution_blocked'
    assert not boundary.state['writes']


def test_core_denial_wins_and_human_token_is_not_forgeable(boundary):
    confirmation=boundary.ticket()
    assert boundary.send(confirmation,'x'*43).json()['status']=='execution_blocked'
    with boundary.desk.db() as db:db.execute("UPDATE tickets SET state='revoked'")
    assert boundary.send(confirmation).json()['status']=='execution_blocked'
    assert not boundary.state['writes']


def test_live_switch_at_core_blocks_valid_preview(boundary):
    confirmation=boundary.ticket()
    boundary.config['dry_run']=False
    assert boundary.send(confirmation).json()['status']=='execution_blocked'
    assert not boundary.state['writes']


def test_core_risk_rejection_cannot_be_overridden(boundary):
    confirmation=boundary.ticket()
    boundary.desk.minimum_stake.return_value=50
    assert boundary.send(confirmation).json()['status']=='execution_blocked'
    assert not boundary.state['writes']


def test_valid_non_live_ticket_delegates_once_and_never_claims_fill(boundary):
    confirmation=boundary.ticket()
    first=boundary.send(confirmation).json()
    assert first['status']=='submitted' and first['execution_authorized'] is True
    assert 'fill' in first['notice']
    assert len(boundary.state['writes'])==1
    assert boundary.state['writes'][0][0]=='forceenter'
    assert boundary.send(confirmation).json()['status']=='submitted'
    assert len(boundary.state['writes'])==1


def test_duplicate_modified_after_submission_is_rejected(boundary):
    confirmation=boundary.ticket()
    assert boundary.send(confirmation).json()['status']=='submitted'
    confirmation['intent']['stake']=30
    assert boundary.send(confirmation).json()['status']=='execution_blocked'
    assert len(boundary.state['writes'])==1


def test_ambiguous_delivery_and_core_restart_never_resubmit(boundary):
    confirmation=boundary.ticket();boundary.state['ambiguous']=True
    result=boundary.send(confirmation).json()
    assert result['status']=='unknown' and result['retry_allowed'] is False
    assert boundary.send(confirmation).json()['status']=='unknown'
    assert len(boundary.state['writes'])==1
    # Recreate the actual durable Core desk facade, without auto-confirming.
    restarted=Desk(boundary.desk.root,transport=boundary.desk.transport)
    result=restarted.confirm(confirmation['ticket'],expected={
        'intent':confirmation['intent'],'identity':confirmation['identity']})
    assert result['state']=='unknown'
    assert len(boundary.state['writes'])==1


def test_position_close_requires_exact_existing_human_ticket(boundary):
    confirmation=boundary.ticket('exit')
    changed=copy.deepcopy(confirmation);changed['intent']['trade_id']=10
    assert boundary.send(changed).json()['status']=='execution_blocked'
    assert boundary.send(confirmation).json()['status']=='submitted'
    assert boundary.state['writes']==[('forceexit',{'tradeid':'9','ordertype':'market'})]


def test_core_unavailable_or_bad_response_never_retries(boundary,monkeypatch):
    original=httpx.Client.__class__  # use a minimal client to avoid real network
    class Unavailable:
        count=0
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def post(self,*args,**kwargs):
            self.count+=1
            raise httpx.ReadTimeout('fixture after submission')
    client=Unavailable()
    monkeypatch.setattr(httpx,'Client',lambda **kwargs:client)
    result=boundary.adapter.confirm_core_ticket(boundary.ticket(),boundary.token)
    assert result['status']=='unknown' and result['retry_allowed'] is False
    assert client.count==1 and not boundary.state['writes']


def test_execution_denial_preserves_read_only_market_analysis(boundary):
    assert boundary.adapter.execute(TradeIntent('BTC/USDT','long',1,100,99,102,.9,True))['status']=='execution_blocked'
    from app.trading.market_adapter import MarketDataAdapter
    from app.trading.market_intelligence import MarketIntelligenceEngine
    market=MarketDataAdapter()
    snapshot=market.ingest_candles('BTC/USDT','1h',[
        (1000+i*3600,100+i,102+i,99+i,101+i,1000) for i in range(60)])
    state=MarketIntelligenceEngine().analyze(snapshot)
    assert state.symbol=='BTC/USDT'
    assert not boundary.state['writes']


@pytest.mark.parametrize('operation',['start_bot','stop_bot','restart_bot','emergency_shutdown'])
def test_supervisor_mutations_fail_closed_without_network(monkeypatch,operation):
    from app.trading.freqtrade_supervisor import FreqtradeSupervisor
    calls=[]
    monkeypatch.setattr('urllib.request.urlopen',lambda *args,**kwargs:calls.append(args))
    result=getattr(FreqtradeSupervisor(),operation)()
    assert result['status']=='execution_blocked'
    assert not calls


def test_supervisor_reads_remain_available(monkeypatch):
    from app.trading.freqtrade_supervisor import FreqtradeSupervisor
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self):return b'{"status":"running"}'
    calls=[]
    def read(request,timeout):
        calls.append(request.get_method());return Response()
    monkeypatch.setattr('urllib.request.urlopen',read)
    assert FreqtradeSupervisor().get_trades()=={'status':'running'}
    assert calls==['GET']


def test_core_original_confirmation_interface_is_unchanged(boundary):
    confirmation=boundary.ticket()
    response=boundary.core.post('/api/confirm',json={
        'ticket':confirmation['ticket'],'acknowledgement':'CONFIRM_SIMULATED_ORDER'},
        headers=boundary.headers)
    assert response.status_code==200 and response.json()['state']=='submitted'
    assert len(boundary.state['writes'])==1




def test_running_human_desk_cannot_bypass_unavailable_bound_core(boundary):
    confirmation=boundary.ticket()
    boundary.core_available[0]=False
    result=boundary.send(confirmation).json()
    assert result["execution_authorized"] is False
    assert result["status"]=="unknown" and result["retry_allowed"] is False
    assert not boundary.state["writes"]
    assert len(boundary.calls)==1  # Core did not forward to the human gateway.
