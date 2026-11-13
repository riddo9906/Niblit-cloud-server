"""Opt-in local shared transport. Existing memory gateway owns every reservation."""
from contextvars import ContextVar
from pathlib import Path
from uuid import uuid4
import os
import httpx
from fastapi import HTTPException

delegated_token=ContextVar('cloud_delegated_token',default=None)

class SharedGGUFEngine:
    def __init__(self, **kwargs): pass
    def chat(self,messages,temperature,max_tokens,tools=None,tool_choice=None):
        from app.main import ModelEngineResult, normalize_tool_calls
        key=Path(os.environ['NIBLIT_CLOUD_GGUF_KEY_FILE']).read_text().strip()
        if len(key)<32:raise HTTPException(503,'Shared model authentication unavailable')
        headers={'Authorization':'Bearer '+key}
        parent=delegated_token.get();owned=parent is None;job=uuid4().hex
        authority='http://127.0.0.1:8772/api/local-tutor/task/'
        def check(response):
            if response.status_code==409:raise HTTPException(409,'Shared inference task busy')
            if response.status_code>=400:raise HTTPException(503,'Shared inference authority rejected or unavailable')
            return response.json()
        with httpx.Client(trust_env=False,timeout=5) as client:
            if owned:parent=check(client.post(authority+'acquire',headers=headers,json={'owner':'core_inference'}))['token']
            claimed=False
            try:
                check(client.post(authority+'delegate',headers=headers,json={'token':parent,'request_id':job}));claimed=True
                body={'model':'qwen2.5-coder-3b','messages':messages,'temperature':temperature,'max_tokens':max_tokens,'stream':False}
                if tools:body['tools']=tools
                if tool_choice is not None:body['tool_choice']=tool_choice
                response=client.post('http://127.0.0.1:8766/v1/chat/completions',headers=headers,json=body,timeout=240)
                response.raise_for_status();data=response.json();choice=data['choices'][0];message=choice['message']
                check(client.post(authority+'delegate',headers=headers,json={'token':parent,'request_id':job,'complete':True}));claimed=False
                if owned:check(client.post(authority+'finish',headers=headers,json={'token':parent}))
                return ModelEngineResult(text=message.get('content') or '',finish_reason=choice.get('finish_reason','stop'),usage=data.get('usage'),tool_calls=normalize_tool_calls(message.get('content'),message.get('tool_calls'),tools))
            except BaseException as exc:
                rejected=isinstance(exc,httpx.HTTPStatusError) and exc.response.status_code in (400,401,403,404,413,422)
                # No retry/fallback: a lost completion may still be generating.
                if claimed:
                    try:client.post(authority+'delegate',headers=headers,json={'token':parent,'request_id':job,'complete':True,'uncertain':not rejected})
                    except Exception:pass
                if owned:
                    try:client.post(authority+'finish',headers=headers,json={'token':parent,'uncertain':not rejected})
                    except Exception:pass
                if rejected:raise HTTPException(exc.response.status_code,'Local model rejected the request') from exc
                raise
