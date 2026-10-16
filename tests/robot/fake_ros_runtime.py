"""Strict, in-memory ROS/SDK boundary doubles. Never contacts a robot."""
import copy
import time
import types


class Vector3:
    def __init__(self): self.x = self.y = self.z = 0.0


class Twist:
    def __init__(self): self.linear = Vector3(); self.angular = Vector3()


class Bool:
    def __init__(self, data=False): self.data = data


class Odometry:
    def __init__(self):
        self.pose = types.SimpleNamespace(pose=types.SimpleNamespace(
            position=Vector3(), orientation=types.SimpleNamespace(x=0., y=0., z=0., w=1.)))
        self.header = stamp_header()


def stamp_header():
    now = time.time_ns()
    return types.SimpleNamespace(stamp=types.SimpleNamespace(sec=now//10**9, nanosec=now%10**9))


class PointCloud2:
    def __init__(self, points): self.points = points; self.header = stamp_header()


class Publisher:
    def __init__(self): self.messages = []
    def publish(self, message): self.messages.append(copy.deepcopy(message))


class Logger:
    def __getattr__(self, name):
        if name not in {'info','warning','warn','error','fatal','debug'}: raise AttributeError(name)
        return lambda *args, **kwargs: None


class Node:
    overrides = {}
    def __init__(self, name):
        self.name = name; self.params = {}; self.publishers = {}; self.subscriptions = {}
    def declare_parameter(self, name, default): self.params[name] = self.overrides.get(name, default)
    def get_parameter(self, name): return types.SimpleNamespace(value=self.params[name])
    def create_publisher(self, message_type, topic, qos):
        pub = Publisher(); self.publishers[topic] = pub; return pub
    def create_subscription(self, message_type, topic, callback, qos):
        self.subscriptions[topic] = callback; return callback
    def create_timer(self, period, callback): return types.SimpleNamespace(period=period, callback=callback)
    def get_logger(self): return Logger()
    def get_clock(self): return types.SimpleNamespace(now=lambda: types.SimpleNamespace(nanoseconds=time.time_ns()))
    def destroy_node(self): return True


class SportClient:
    instances = []
    def __init__(self, enableLease=False):
        assert enableLease is True
        self.calls=[]; self.instances.append(self)
    def SetTimeout(self, timeout): assert timeout == 3.0
    def Init(self): return None
    def WaitLeaseApplied(self): return 0
    def StopMove(self): self.calls.append(('stop',)); return 0
    def Move(self, vx, vy, wz): self.calls.append(('move',vx,vy,wz)); return 0


def install(monkeypatch):
    def module(name, **attrs):
        obj=types.ModuleType(name); obj.__path__=[]
        for key,value in attrs.items():setattr(obj,key,value)
        monkeypatch.setitem(__import__('sys').modules,name,obj)
        return obj
    module('rclpy', ok=lambda: True, init=lambda: None, shutdown=lambda: None)
    module('rclpy.node', Node=Node)
    module('rclpy._rclpy_pybind11', RCLError=type('RCLError',(RuntimeError,),{}))
    module('rclpy.executors', ExternalShutdownException=type('ExternalShutdownException',(Exception,),{}))
    module('rclpy.signals', SignalHandlerOptions=types.SimpleNamespace(NO=0))
    module('rclpy.qos',qos_profile_sensor_data=object())
    module('geometry_msgs');module('geometry_msgs.msg',Twist=Twist)
    module('std_msgs');module('std_msgs.msg',Bool=Bool)
    module('nav_msgs');module('nav_msgs.msg',Odometry=Odometry)
    module('sensor_msgs');module('sensor_msgs.msg',PointCloud2=PointCloud2)
    cloud=module('sensor_msgs_py.point_cloud2',read_points=lambda msg,**kwargs:iter(msg.points))
    module('sensor_msgs_py',point_cloud2=cloud)
    module('unitree_sdk2py');module('unitree_sdk2py.core')
    module('unitree_sdk2py.core.channel',ChannelFactoryInitialize=lambda domain,interface:None)
    module('unitree_sdk2py.go2');module('unitree_sdk2py.go2.sport')
    module('unitree_sdk2py.go2.sport.sport_client',SportClient=SportClient)
