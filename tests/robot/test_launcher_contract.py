"""Run the real Bash launcher with inert processes and inspect actual argv."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import ast
import argparse
import math

import pytest

ROOT=Path(__file__).resolve().parents[2]


def bash_path():
    if os.name=='nt':
        git=shutil.which('git')
        if git:
            path=Path(git).parent.parent/'bin/bash.exe'
            if path.is_file():return str(path)
    path=shutil.which('bash')
    if not path:pytest.skip('Bash is required for launcher integration')
    return path


def script(path,text):
    path.write_bytes(text.encode());path.chmod(0o755);return str(path).replace('\\','/')


@pytest.mark.parametrize('mode,manual',[('shadow',False),('live',False),('live',True)])
def test_real_launcher_argv_is_valid_and_no_implicit_arm(tmp_path,mode,manual):
    tools=tmp_path/'tools';tools.mkdir();log=tmp_path/'argv.log'
    fake_python=script(tools/'python.sh','''#!/usr/bin/env bash
printf '%s\\037' "$@" > "$FAKE_ARGV_LOG.$$"
printf '\\n' >> "$FAKE_ARGV_LOG.$$"
exit 0
''')
    script(tools/'setsid','#!/usr/bin/env bash\nexec "$@"\n')
    script(tools/'pgrep','#!/usr/bin/env bash\nexit 1\n')
    # Exercise the exact unset-variable pattern in ROS setup under set -u.
    ros=script(tmp_path/'ros.bash','export MOCK_ROS_OPTIONAL="$MOCK_UNSET_TRACE_VAR"\n')
    stop=tmp_path/'STOP';arm=tmp_path/'ARM';stop.touch()
    goal=tmp_path/'goal.json';goal.write_text('{}')  # preflight is inert here
    env=os.environ.copy();env.update({
        'PATH':str(tools)+os.pathsep+env['PATH'],
        'V70_PYTHON':fake_python,'V70_ROS_SETUP':ros,'V70_SENSOR_SETUP':'',
        'V70_OUTPUT_DIR':str(tmp_path/'run'),'GO2_NAV_STOP_FILE':str(stop),
        'GO2_MOTION_ARM_FILE':str(arm),'V70_OPERATOR_POSTURE':'standing',
        'V70_OPERATOR_STOP_ONLY':'1' if manual else '0',
        'V70_OPERATOR_STOP_ACK':'I_WILL_STOP_THE_ROBOT_MANUALLY' if manual else '',
        'V70_INTERFACE':'fake0','V70_ODOM_TOPIC':'/test/odom','V70_CLOUD_TOPIC':'/test/cloud',
        'FAKE_ARGV_LOG':str(log),'MSYS2_ARG_CONV_EXCL':'*',
    })
    result=subprocess.run([bash_path(),str(ROOT/'robot/run_go2.bash'),mode,str(goal)],
                          cwd=ROOT,env=env,text=True,capture_output=True,timeout=15)
    assert result.returncode==0,(result.stdout,result.stderr)
    calls=[line.rstrip('\x1f').split('\x1f') for file in tmp_path.glob('argv.log.*')
           for line in file.read_text().splitlines()]
    planner=next(c for c in calls if c[0].endswith('v70_live_nav_fast.py'))
    assert planner[planner.index('--interface')+1]=='fake0'
    assert planner[planner.index('--odom-topic')+1]=='/test/odom'
    assert ('--live' in planner)==(mode=='live')
    if mode=='live':
        assert '--async-plan-images' in planner and '--save-plan-images' in planner
        assert ('--operator-stop-only' in planner)==manual
    else:assert '--shadow-once' in planner
    executor=next(c for c in calls if c[0].endswith('motion_executor_v2.py'))
    assert 'interface:=fake0' in executor
    assert 'planned_neutral_move:=false' in executor
    assert not arm.exists() and stop.exists()
    # Parse and validate the captured argv with the production implementation,
    # without importing ROS or instantiating any control client.
    source=ast.parse((ROOT/'deploy/v70_go2_orin/v70_live_nav_fast.py').read_text(encoding='utf-8'))
    functions=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in {'parse_args','validate_args'}]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+functions,type_ignores=[])
    ast.fix_missing_locations(module)
    sys.path.insert(0,str(ROOT/'deploy/v70_go2_orin'))
    from v70_closed_loop_core import active_history_scan_targets
    namespace={'argparse':argparse,'math':math,'Path':Path,
               'active_history_scan_targets':active_history_scan_targets}
    exec(compile(module,'production_launcher_arguments','exec'),namespace)
    previous=sys.argv
    try:
        sys.argv=planner
        namespace['validate_args'](namespace['parse_args']())
    finally:sys.argv=previous
