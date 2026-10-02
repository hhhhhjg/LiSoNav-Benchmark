"""CLI runner with dynamic agent loading and incremental JSON result storage."""
import argparse
import importlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from .api import NavigationEnvironment, BenchmarkConfig
from .data import Dataset

def save_json(path, value):
    """Publish complete JSON atomically; represent missing distances as null."""
    import math
    def clean(v):
        if isinstance(v,float) and not math.isfinite(v):
            return None
        if isinstance(v,dict):
            return {k:clean(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)):
            return [clean(x) for x in v]
        return v
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(tmp,path)

def main():
    p = argparse.ArgumentParser(description='LiSoNav agent independent evaluation')
    p.add_argument('--dataset')
    p.add_argument('--assets')
    p.add_argument('--backend',choices=['mock','habitat'],default='habitat')
    p.add_argument('--agent',default='agents.mock_agent:MockAgent')
    p.add_argument('--output',default='results/mock_agent')
    p.add_argument('--limit-sequences',type=int)
    p.add_argument('--limit-tasks',type=int)
    p.add_argument('--path-budget-m',type=float)
    p.add_argument('--path-budget-multiplier',type=float,default=10.0)
    p.add_argument('--max-actions',type=int,default=1000)
    p.add_argument('--width',type=int,default=1280)
    p.add_argument('--height',type=int,default=1280)
    p.add_argument('--gpu-device-id',type=int,default=-1)
    p.add_argument('--no-reference-images',action='store_true')
    args = p.parse_args()
    for name in ['limit_sequences','limit_tasks']:
        if getattr(args,name) is not None and getattr(args,name)<=0:
            p.error(f'--{name.replace("_","-")} must be positive')
    dataset = Dataset(args.dataset,args.assets,not args.no_reference_images)
    config = BenchmarkConfig(path_budget_m=args.path_budget_m,path_budget_multiplier=args.path_budget_multiplier,
        max_actions=args.max_actions,width=args.width,height=args.height,gpu_device_id=args.gpu_device_id)
    module, cls = args.agent.split(':')
    agent = getattr(importlib.import_module(module),cls)()
    output = Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    if (output/'results.json').exists():
        p.error(f'{output}/results.json exists; choose a new output directory')
    env = NavigationEnvironment(dataset,config,args.backend)
    results = []
    save_json(output/'run_config.json',dict(config=asdict(config),arguments=vars(args),dataset=str(dataset.root),assets=str(dataset.assets)))
    try:
        for seq_index,sequence in enumerate(dataset.sequences()):
            if args.limit_sequences is not None and seq_index>=args.limit_sequences:
                break
            env.begin_sequence(sequence)
            for episode in sequence.episodes:
                env.begin_episode(episode)
                for subtask in episode.subtasks:
                    task, observation = env.reset_task(subtask)
                    agent.reset(task)
                    while not observation.done:
                        observation = env.step(agent.act(observation))
                    result = env.evaluate()
                    # Agent callbacks receive no hidden GT coordinates or semantic history.
                    agent.on_task_end({key: result[key] for key in [
                        'subtask_id','stop_reason','forced_stop','explored_distance',
                        'action_count','success_by_distance','spl_by_distance','collision_ratio']})
                    results.append(result)
                    save_json(output/'results.json',results)
                    save_json(output/'summary.json',env.aggregate())
                    history_dir = output / f'{sequence.scene_dir.name}__{sequence.scene_id}_ep_{sequence.episode_id}'
                    history_dir.mkdir(exist_ok=True)
                    save_json(history_dir/'gt_history_retrieval_state.json',env.history_state())
                    print(f"{task.subtask_id}: {result['stop_reason']}, path={result['explored_distance']:.3f}m, SR={result['success_by_distance']}",flush=True)
                    if args.limit_tasks is not None and len(results)>=args.limit_tasks:
                        return
    finally:
        env.close()
        agent.close()

if __name__=='__main__':
    main()
