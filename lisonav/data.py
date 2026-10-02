"""Dataset interface. Only TaskRequest is passed to an agent; GT stays here."""
import json
from pathlib import Path
import numpy as np
from .dataset import discover_scene_dirs, load_runtime_bundle, build_episodes
from .types import TaskRequest

ROOT = Path(__file__).resolve().parents[1]

class Dataset:
    def __init__(self, root=None, assets=None, reference_images=True):
        self.root = Path(root or ROOT/'LiSoNav-Eval/lifelong_navigation_sequences/part00').resolve()
        self.assets = Path(assets or ROOT/'LiSoNav-Eval/assets/data').resolve()
        self.reference_images = reference_images
        self._surface_descriptions = {}
        path = self.assets/'ycb_and_hssd/name2description.json'
        self.descriptions = json.loads(path.read_text()) if path.exists() else {}
        self.scene_dirs = discover_scene_dirs(self.root)
        if not self.scene_dirs:
            raise ValueError(f'No navigation sequences found in {self.root}')

    def sequences(self):
        for scene_dir in self.scene_dirs:
            bundle = load_runtime_bundle(scene_dir)
            for index in range(len(bundle['task_data']['episodes'])):
                sequence = build_episodes(scene_dir, bundle, index)
                expected = len(bundle['task_data']['episodes'][index].get('tasks', []))
                actual = sum(len(episode.subtasks) for episode in sequence.episodes)
                if actual != expected:
                    raise ValueError(f'{scene_dir}: loaded {actual}/{expected} tasks; missing or invalid goals')
                yield sequence

    def task_request(self, sequence, episode, subtask, *, new_sequence, new_episode):
        description = self.descriptions.get(subtask.goal_category, subtask.goal_category.replace('_', ' '))
        reference = None
        image = self.assets/'ycb_and_hssd/views_center_640'/f'{subtask.goal_category}.png'
        if self.reference_images and image.exists():
            from PIL import Image
            with Image.open(image) as im:
                reference = np.asarray(im.convert('RGB')).copy()
        lang = subtask.goal_entries[0].get('lang_desc', description)
        instruction = (f'Can you find the {description}?' if subtask.goal_type == 'object'
                       else f'Find the object described as: {lang}' if subtask.goal_type == 'description'
                       else 'Find the object shown in the reference image.')
        sequence_id = f'{sequence.scene_dir.name}__{sequence.episode_id}'
        if sequence.scene_dir not in self._surface_descriptions:
            path = sequence.scene_dir/'navigation_task_pipeline/target_surface_final_descriptions_qwen3vlplus.json'
            self._surface_descriptions[sequence.scene_dir] = json.loads(path.read_text()).get('descriptions_by_subtask', {}) if path.exists() else {}
        surface = self._surface_descriptions[sequence.scene_dir].get(f'target_{subtask.task_index:02d}', '')
        return TaskRequest(
            f'{sequence.scene_dir.name}__{sequence.scene_id}_{sequence.episode_id}_e{episode.episode_index}_s{episode.subtasks.index(subtask)}',
            sequence_id, sequence.scene_id, episode.episode_index, subtask.task_index,
            subtask.goal_type, subtask.goal_category, instruction, description, surface, reference,
            new_sequence, new_episode)
