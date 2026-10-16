"""Exercise production ROS nodes with simulated boundaries, not a real robot."""
import importlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'deploy/v70_go2_orin'))
sys.path.insert(0,str(Path(__file__).parent))
from fake_ros_runtime import install, Node, SportClient, Twist, Bool, Odometry, PointCloud2


@pytest.fixture
def ros(monkeypatch):
    install(monkeypatch)
    Node.overrides={};SportClient.instances=[]
    names=['v70_live_nav_fast','safe_bridge_v5','motion_executor_v2']
    prior={n:sys.modules.pop(n,None) for n in names}
    modules=[importlib.import_module(n) for n in names]
    yield SimpleNamespace(nav=modules[0],bridge=modules[1],executor=modules[2])
    for n in names:
        sys.modules.pop(n,None)
        if prior[n] is not None:sys.modules[n]=prior[n]
    Node.overrides={}


def test_bridge_executor_stop_disarm_stale_and_no_old_velocity(ros,tmp_path):
    stop=tmp_path/'STOP';arm=tmp_path/'ARM';fault=tmp_path/'FAULT';stop.touch()
    Node.overrides={'dry_run':False,'stop_file':str(stop),'arm_file':str(arm),
                    'fault_file':str(fault),'max_vx':.30,'max_ax':.75,
                    'clear_recovery_s':.01,'min_clear_observations':1,
                    'planned_neutral_move':False,'neutral_hold_s':0.}
    bridge=ros.bridge.SafeBridgeV5();executor=ros.executor.MotionExecutorV2()
    client=executor.client
    points=[(float(x),float(y),-.4) for x in np.linspace(.35,1.3,10) for y in np.linspace(-.2,.2,7)]
    def sensors():
        bridge.odom_callback(Odometry());bridge.cloud_callback(PointCloud2(points))
    def deliver(vx,enable=True):
        msg=Twist();msg.linear.x=vx;bridge.cmd_callback(msg);sensors();bridge.control_timer()
        executor.enable_callback(Bool(enable));executor.command_callback(bridge.publisher.messages[-1]);executor.control_timer()
    deliver(.2)
    assert not any(c[0]=='move' for c in client.calls)
    time.sleep(.02)
    arm.touch();stop.unlink();deliver(.2)
    assert any(c[0]=='move' and c[1]>0 for c in client.calls)
    count=sum(c[0]=='move' and c[1]>0 for c in client.calls)
    deliver(0.,False)
    assert sum(c[0]=='move' and c[1]>0 for c in client.calls)==count
    assert client.calls[-1][0]=='stop'
    # A delayed nonzero safe command from before enable rising-edge cannot restart.
    msg=Twist();msg.linear.x=.2;executor.command_callback(msg)
    executor.enable_callback(Bool(True));executor.control_timer()
    assert sum(c[0]=='move' and c[1]>0 for c in client.calls)==count
    deliver(.2);arm.unlink();executor.control_timer()
    assert client.calls[-1][0]=='stop'
    arm.touch();deliver(.2);executor.last_cmd_time=time.monotonic()-2.;executor.control_timer()
    assert client.calls[-1][0]=='stop'
    stop.touch();deliver(.2)
    assert client.calls[-1][0]=='stop'


class Camera:
    def __init__(self,nav): self.nav=nav;self.images=[Image.new('RGB',(64,64),(30+i,60,80)) for i in range(4)]
    def snapshot(self,**kwargs):
        now=time.monotonic()
        return [self.nav.ImageSample(now-(3-i)*.2,img,str(i),64,64) for i,img in enumerate(self.images)]
    def diagnostics(self):return {'sample_count':4,'latest_age_s':0.,'last_error':None}
    def stop(self):pass


def test_production_live_node_initializes_and_shuts_down(ros,tmp_path,monkeypatch):
    nav=ros.nav
    goal=tmp_path/'goal.jpg';Image.new('RGB',(64,64),(10,20,30)).save(goal)
    route_data=json.loads((ROOT/'robot/goal.example.json').read_text(encoding='utf-8'))
    route_data['goals'][0]['image']=str(goal);routefile=tmp_path/'goal.json';routefile.write_text(json.dumps(route_data))
    (tmp_path/'STOP').touch()
    monkeypatch.setattr(sys,'argv',['v70_live_nav_fast.py','--route-manifest',str(routefile),
        '--live','--operator-stop-only','--cem-iterations','1','--live-max-vx','0.30',
        '--output-dir',str(tmp_path/'run'),
        '--stop-file',str(tmp_path/'STOP'),'--arm-file',str(tmp_path/'ARM'),
        '--ready-file',str(tmp_path/'READY')])
    args=nav.parse_args();nav.validate_args(args)
    route=nav.RouteManifest.load(routefile,require_calibrated=False)
    planner=SimpleNamespace(load_timings_ms={},summary=lambda result:dict(result))
    node=nav.V70LiveNavNode(args=args,planner=planner,camera=Camera(nav),route=route)
    try:
        assert node.fresh_plan_gate.last_consumed_sequence==-1
        assert not node.active_history_scan_enabled
        node._publish_zero()
        assert node.cmd_publisher.messages[-1].linear.x==0.
        assert node.enable_publisher.messages[-1].data is False
        node.odom.append(Odometry())
        # No valid observations/planning output: the actual control tick must stay zero.
        node._control_timer()
        assert node.cmd_publisher.messages[-1].linear.x==0.
        # Feed the production plan handler a valid new plan, using the same
        # DTOs as its actual worker. Only the learned output is synthetic here.
        (tmp_path/'ARM').touch();(tmp_path/'STOP').unlink()
        node.ready=True;node.last_interlock=True;node.interlock_generation=1
        def fill_pose(x):
            node.odom.samples.clear();now=time.monotonic()
            for i in range(6):node.odom.samples.append(nav.PoseSample(now-(5-i)*.2,x,0.,0.))
        def make_output(sequence,abstained,x=0.):
            now=time.monotonic();images=tuple(node.camera.images)
            action=np.array([.12,0.,0.],dtype=np.float32)
            actions=np.tile(action,(4,1)) if not abstained else np.zeros((4,3),dtype=np.float32)
            result={'abstained':abstained,'selection_reason':'finite_safe_fixed_pool',
                'selected_first_action':actions[0], 'selected_actions_h4':actions,
                'dino':{'current_goal_cosine_distance':.7},'nomad':{'temporal_distance':15.}}
            def summary(r):
                return {'abstained':r['abstained'],'selection_reason':r['selection_reason'],
                    'selected_first_action':r['selected_first_action'].tolist(),
                    'selected_actions_h4':r['selected_actions_h4'].tolist(),
                    'selected_waypoints_h4_m':np.cumsum(r['selected_actions_h4'][:,:2],axis=0).tolist()}
            node.planner.summary=summary
            job=nav.PlanningJob(sequence=sequence,route_index=0,goal_name='your-target',
                goal_path=goal,goal_sha256=nav.sha256_file(goal),observations=images,
                image_sha256=('0','1','2','3'),poses=tuple(node.odom.samples)[-4:],
                visual_observations=images,visual_image_sha256=('0','1','2','3'),
                visual_poses=tuple(node.odom.samples)[-4:],history_source='fresh_endpoint',
                snapshot_time=now-.005,snapshot_pose=nav.Pose2D(now-.005,x,0.,0.),
                interlock_generation=1,motion_authorized_at_snapshot=True,
                route_intent_direction=0,operator_visual_hint=None,operator_direction_prior=0,
                settle_prefetch=False,motion_prefetch=False)
            return nav.PlanningOutput(job=job,result=result,completed_time=now)
        fill_pose(0.)
        node._handle_plan_output(make_output(0,False),time.monotonic())
        assert node.follower.active, node.last_plan_summary
        assert node.fresh_plan_gate.last_consumed_sequence==0
        assert node.pending_execution_actions==[]
        node._control_timer()
        assert node.cmd_publisher.messages[-1].linear.x>0., (node.last_command,node.fault_reason,node.last_sensor_error)
        # Completing H1 must hold at zero; a rejected next plan must not replay it.
        node.follower.reset('verified_endpoint')
        fill_pose(.12)
        node._handle_plan_output(make_output(1,True,.12),time.monotonic())
        assert not node.follower.active
        assert node.cmd_publisher.messages[-1].linear.x==0.
        node._handle_plan_output(make_output(0,False,.12),time.monotonic())
        assert not node.follower.active
        assert node.fresh_plan_gate.last_consumed_sequence==0
        assert node.cmd_publisher.messages[-1].linear.x==0.
        # Visual-only evidence cannot restore a cached, unexecuted old plan.
        cached=make_output(2,False,.12)
        cached.result['visual_recheck_only']=True
        node.target_visual_recheck_plan=cached;node.target_visual_recheck_active=True
        assert node._handle_target_visual_recheck(output=cached,result=cached.result,
            latest=node.odom.latest(),stationary=True,drift_m=0.,drift_yaw=0.,now=time.monotonic())
        assert node.target_visual_recheck_plan is None
        assert not node.target_visual_recheck_active and not node.follower.active
        assert node.fresh_plan_gate.last_consumed_sequence==0
        assert node.cmd_publisher.messages[-1].linear.x==0.
    finally:
        node.shutdown();node.destroy_node()
    assert (tmp_path/'STOP').exists()
