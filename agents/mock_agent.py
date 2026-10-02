"""Replace this class or supply --agent your_module:YourAgent."""
from lisonav.types import Action

class MockAgent:
    def reset(self, task):
        self.task = task
        self.index = 0
        if task.new_sequence:
            self.memory = []

    def act(self, observation):
        actions = [Action('move_forward', 1.0), Action('turn_right', 90.0),
                   Action('move_forward', 1.0), Action('stop')]
        action = actions[min(self.index, len(actions)-1)]
        self.index += 1
        return action

    def on_task_end(self, result):
        self.memory.append(result['subtask_id'])

    def close(self):
        pass
