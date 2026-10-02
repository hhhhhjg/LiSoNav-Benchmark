"""Simulator adapters. Neither imports a navigation policy or model weights."""
import json
import math
from pathlib import Path
import numpy as np

class MockSimulator:
    """Open plane with synthetic RGB-D, for testing the API without Habitat."""
    pathfinder = None
    def __init__(self, assets, config):
        self.config = config
    def load_episode(self, sequence, episode):
        state = episode.start_agent_state
        self.position = np.array(state['position'], dtype=float)
        self.rotation = np.array(state['rotation'], dtype=float)
        x, y, z, w = self.rotation
        self.heading = math.atan2(2*(w*y+x*z), 1-2*(y*y+z*z))
    def execute(self, name, amount):
        if name == 'move_forward':
            self.position += np.array([-math.sin(self.heading), 0, -math.cos(self.heading)]) * amount
        else:
            self.heading += math.radians(amount) * (-1 if name == 'turn_right' else 1)
            self.rotation = np.array([0, math.sin(self.heading/2), 0, math.cos(self.heading/2)])
        return False
    def capture(self):
        h, w = self.config.height, self.config.width
        c, s = math.cos(self.heading), math.sin(self.heading)
        pose = np.eye(4)
        tilt = math.radians(self.config.camera_tilt_deg)
        ct, st = math.cos(tilt), math.sin(tilt)
        pose[:3,:3] = np.array([[c,0,s],[0,1,0],[-s,0,c]]) @ np.array([[1,0,0],[0,ct,-st],[0,st,ct]])
        pose[:3,3] = self.position + [0,self.config.camera_height,0]
        return dict(rgb=np.zeros((h,w,3),np.uint8), depth=np.ones((h,w),np.float32),
                    semantic=np.zeros((h,w),np.int32), position=self.position.copy(),
                    rotation_xyzw=self.rotation.copy(), camera_pose=pose)
    def close(self):
        pass

class HabitatSimulator:
    """Habitat-Sim scene, object-layout, sensor and discrete action adapter."""
    def __init__(self, assets, config):
        import habitat_sim
        self.hs = habitat_sim
        self.assets = Path(assets)
        self.config = config
        self.sim = None
        self.scene_id = None
        self.pathfinder = None
        self.object_ids = []

    def _open_scene(self, sequence):
        hs, cfg = self.hs, self.config
        self.close()
        scene_path = Path(sequence.scene_path)
        # Released annotations use hm3d/val/... relative to assets/data.
        scene_path = self.assets / scene_path if not scene_path.is_absolute() else scene_path
        if not scene_path.exists():
            raise FileNotFoundError(f'HM3D scene missing: {scene_path}')
        sim_cfg = hs.SimulatorConfiguration()
        sim_cfg.scene_id = str(scene_path)
        dataset_path = Path(sequence.scene_dataset_config)
        sim_cfg.scene_dataset_config_file = str(self.assets / dataset_path if not dataset_path.is_absolute() else dataset_path)
        sim_cfg.load_semantic_mesh = True
        sim_cfg.gpu_device_id = cfg.gpu_device_id
        sim_cfg.navmesh_settings = self._navmesh_settings()
        agent_cfg = hs.agent.AgentConfiguration()
        agent_cfg.radius = 0.001
        agent_cfg.height = cfg.agent_height
        sensors = []
        for uuid, kind in [('rgb',hs.SensorType.COLOR), ('depth',hs.SensorType.DEPTH), ('semantic',hs.SensorType.SEMANTIC)]:
            spec = hs.CameraSensorSpec()
            spec.uuid = uuid
            spec.sensor_type = kind
            spec.resolution = [cfg.height,cfg.width]
            spec.position = [0,cfg.camera_height,0]
            spec.orientation = [math.radians(cfg.camera_tilt_deg),0,0]
            spec.hfov = cfg.hfov
            sensors.append(spec)
        agent_cfg.sensor_specifications = sensors
        agent_cfg.action_space = {
            name: hs.agent.ActionSpec(name, hs.agent.ActuationSpec(amount=0.25 if name=='move_forward' else 90.0))
            for name in ['move_forward','turn_right','turn_left']
        }
        self.sim = hs.Simulator(hs.Configuration(sim_cfg,[agent_cfg]))
        self.scene_id = sequence.scene_id
        self.agent = self.sim.initialize_agent(0)
        self.pathfinder = self.sim.pathfinder
        self.pathfinder.seed(cfg.seed)
        self.templates = self.sim.get_object_template_manager()
        configs = self.assets/'ycb_and_hssd/configs'
        if not configs.exists():
            raise FileNotFoundError(f'Object templates missing: {configs}')
        self.templates.load_configs(str(configs))
        for handle in sorted(self.templates.get_file_template_handles()):
            key = Path(handle).name.split('.')[0]
            self.templates.register_template(self.templates.get_template_by_handle(handle),key)
        self.objects = self.sim.get_rigid_object_manager()

    def _navmesh_settings(self):
        settings = self.hs.NavMeshSettings()
        settings.set_defaults()
        settings.agent_radius = self.config.navmesh_agent_radius
        settings.agent_height = self.config.agent_height
        settings.include_static_objects = True
        return settings

    def load_episode(self, sequence, episode):
        if self.scene_id != sequence.scene_id:
            self._open_scene(sequence)
        import magnum as mn
        from habitat_sim.utils.common import quat_from_coeffs
        for object_id in self.object_ids:
            self.objects.remove_object_by_id(object_id)
        self.object_ids = []
        layout = json.loads(episode.scene_config_path.read_text())
        for item in layout['objects']:
            sid = int(item['semantic_id'])
            handle = layout['id_handle_mapping'][str(sid)]
            template = self.templates.get_template_by_handle(handle)
            if template is None:
                raise ValueError(f'Missing template {handle} in {episode.scene_config_path}')
            template.semantic_id = sid
            self.templates.register_template(template,handle)
            obj = self.objects.add_object_by_template_handle(handle)
            if obj is None:
                raise RuntimeError(f'Failed to load {handle}')
            self.object_ids.append(obj.object_id)
            obj.translation = mn.Vector3(*item['translation'])
            rot = item.get('rotation',[0,0,0,1])
            obj.rotation = mn.Quaternion(mn.Vector3(*rot[:3]), rot[3])
            obj.motion_type = self.hs.physics.MotionType.KINEMATIC
        if not self.sim.recompute_navmesh(self.pathfinder,self._navmesh_settings()):
            raise RuntimeError(f'Cannot rebuild navmesh for {episode.scene_config_path}')
        state = self.hs.AgentState()
        position = np.asarray(episode.start_agent_state['position'],dtype=np.float32)
        if not self.pathfinder.is_navigable(position):
            position = self.pathfinder.snap_point(position)
        if not np.all(np.isfinite(position)):
            raise ValueError('Episode start cannot be projected onto navmesh')
        state.position = position
        state.rotation = quat_from_coeffs(episode.start_agent_state['rotation'])
        self.agent.set_state(state)

    def execute(self, name, amount):
        self.agent.agent_config.action_space[name].actuation.amount = amount
        observations = self.sim.step(name)
        return bool(observations.get('collided', False))

    def capture(self):
        import quaternion
        from habitat_sim.utils.common import quat_to_coeffs
        raw = self.sim.get_sensor_observations()
        state = self.agent.get_state()
        sensor = state.sensor_states['depth']
        pose = np.eye(4)
        pose[:3,:3] = quaternion.as_rotation_matrix(sensor.rotation)
        pose[:3,3] = sensor.position
        return dict(rgb=raw['rgb'][...,:3].copy(),depth=raw['depth'].copy(),
                    semantic=raw['semantic'].copy(),position=state.position.copy(),
                    rotation_xyzw=np.asarray(quat_to_coeffs(state.rotation)),camera_pose=pose)

    def close(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None
            self.scene_id = None
            self.pathfinder = None
            self.object_ids = []
