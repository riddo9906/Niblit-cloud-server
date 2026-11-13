
"""Run the actual sibling Core/Cloud contract in an isolated interpreter.

Both repositories have flat app/tools packages. Process isolation preserves each
repository's import authority without copying source or creating another build.
"""
from pathlib import Path
import os
import subprocess
import sys
import xml.etree.ElementTree as ET


def test_existing_core_cloud_execution_contract(tmp_path):
    root=Path(__file__).resolve().parents[1]
    cases=Path(__file__).with_name('core_execution_contract_cases.py')
    command=[sys.executable,'-m','pytest',str(cases),'-q','-p','no:cacheprovider',
        '--basetemp='+str(tmp_path/'cases'),
        '--junitxml='+str(tmp_path/'contract.xml')]
    env=dict(os.environ)
    env["NIBLIT_MEMORY_PATH"]=str(tmp_path/"import-memory.json")
    env["NIBLIT_MEMORY_LAB_ENABLED"]="0"
    result=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,timeout=180,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    (tmp_path/'contract.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    assert result.returncode==0,(result.stdout+result.stderr)[-12000:]
    suites=ET.parse(tmp_path/'contract.xml').getroot().iter('testsuite')
    counts=[(int(s.get('tests','0')),int(s.get('failures','0')),int(s.get('errors','0'))) for s in suites]
    assert sum(t for t,_,_ in counts)==34
    assert sum(f+e for _,f,e in counts)==0

