"""Обучение или загрузка construction-политики; сохранение результатов."""
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from tndp.config import ProblemConfig, PPOConfig
from tndp.costs.custom import native_objective, load_cost_function, save_cost_function
from tndp.utils.checkpoints import (
    check_training_destination,
    save_model_weights,
    load_model_weights,
    check_output_writable,
)
from tndp.costs.objective import network_cost
from tndp.utils.random import seed_everything
from tndp.environment.construction import ConstructionEnv
from tndp.costs.preferences import PreferenceSampler
from tndp.training.settings import sampler_params, DEVICE, components, construction_config
from tndp.utils.logging import TensorBoardLogger, ExperimentLogger
from tndp.training.ppo import AlgorithmParams, PPO
from tndp.training.batches import PreferenceDistribution, ConstructionData, EpisodePreferences
from tndp.models.critic import CriticParams, ValueHead
from tndp.training.loop import Budget, train
from tndp.data.transforms import GraphView, pipeline
from tndp.training.validation import ConstructionValidation, EvaluationParams, EvaluationProfile
from tndp.search.policy import ConstructionPolicy
from tndp.data.graph import city_graph
from tndp.data.dataset import save_datasets

UPSTREAM_COMMIT = 'd3b44117a4ea3f0ddfde077386eb6cac2a5eddb8'

def train_gat(train_cities, val_cities, cfg, rl, output_dir, *, tensorboard_dir=None,
              cost_function=network_cost):
    """Train the construction policy with its native batched PPO loop."""
    check_training_destination(output_dir)
    if len(train_cities) < rl.batch_size or not len(val_cities):
        raise ValueError('GAT needs a full city batch and nonempty validation')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    native = construction_config(cfg, rl)
    policy, cost, limits = components(native, cfg)
    save_cost_function(output_dir, cost_function)
    cost = native_objective(cost, cfg, cost_function)
    train_graphs = [city_graph(city) for city in tqdm(train_cities, desc='Подготовка обучения', unit='city', leave=False)]
    val_graphs = [city_graph(city) for city in tqdm(val_cities, desc='Подготовка валидации', unit='city', leave=False)]
    training = GraphView(train_graphs, range(len(train_graphs)), pipeline(native.dataset.train_pipeline))
    validation_graphs = GraphView(val_graphs, range(len(val_graphs)), pipeline(native.dataset.evaluation_pipeline))
    sampler = PreferenceSampler(sampler_params(native.preferences))
    preferences = EpisodePreferences(sampler, PreferenceDistribution.SAMPLER, False)
    val_batch = min(rl.batch_size * 4, len(val_cities))
    data = ConstructionData(
        DataLoader(training, batch_size=rl.batch_size, shuffle=True, drop_last=True),
        DataLoader(validation_graphs, batch_size=val_batch), DEVICE,
        n_routes=(cfg.n_routes,), preferences=preferences, monitor_graphs=validation_graphs,
        validation_indices=list(range(len(val_cities))), validation_batch_size=val_batch,
        test_graphs=None, full_validation=validation_graphs,
        training_source_ids=tuple(range(len(train_cities))),
        validation_source_ids=tuple(range(len(train_cities), len(train_cities) + len(val_cities))),
        test_source_ids=(), online=False,
    )
    env = ConstructionEnv(cost, policy.feature_set, limits, policy.action_spec, node_normalizer=policy.node_norm)
    values = OmegaConf.to_container(native.algorithm, resolve=True)
    critic_params = CriticParams(**values.pop('critic'))
    algorithm = PPO(AlgorithmParams(**values), cost)
    critic = ValueHead(algorithm.value_names, policy.feature_set, critic_params, DEVICE, scalar=algorithm.scalar_values)
    optimizer = torch.optim.Adam(policy.parameters(), lr=native.optimizer.learning_rate,
                                 weight_decay=native.optimizer.decay, maximize=True)
    evaluation = EvaluationParams(**OmegaConf.to_container(native.evaluation, resolve=True))
    profile = EvaluationProfile(evaluation, limits, rl.batch_size, True, ConstructionEnv)
    val_env = ConstructionEnv(cost, policy.feature_set, limits, policy.action_spec, node_normalizer=policy.node_norm)
    validation = ConstructionValidation(data, cost, val_env, profile, DEVICE)
    budget = Budget(rl.total_updates, rl.horizon, native.train_loop.val_period, True,
                    None, 0, None, False, None, None, output_dir, 'gat')
    metadata = dict(backend='connectpt', upstream_commit=UPSTREAM_COMMIT,
                    problem=asdict(cfg), ppo=asdict(rl), architecture='gat',
                    serial_halting=bool(native.model.route_generator.kwargs.serial_halting),
                    native=OmegaConf.to_container(native, resolve=True),
                    validation_decoding='greedy' if rl.validation_deterministic else 'stochastic_1',
                    budget_unit='ppo_updates', timesteps=0, training_seconds=0., status='running')
    config_path = output_dir / 'config.json'
    config_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    started = perf_counter()
    with tqdm(total=rl.total_updates, desc='gat', unit='iter') as progress:
        logs = [ExperimentLogger(output_dir, rl, progress, cfg.alpha),
                TensorBoardLogger(Path(tensorboard_dir) / 'gat' if tensorboard_dir else output_dir)]
        try:
            result = train(env, policy, algorithm, critic, optimizer, logs, data, budget, validation=validation)
            save_model_weights(policy, output_dir / 'last.pt', metadata['native'])
            save_model_weights(result.best_policy, output_dir / 'best.pt', metadata['native'])
            torch.save(result.critic_snapshot, output_dir / 'critic_snapshot.pt')
            metadata.update(timesteps=result.iterations * rl.rollout_size, status='complete',
                            best_validation_cost=result.best_cost)
        finally:
            for logger in logs:
                logger.close()
            metadata['timesteps'] = progress.n * rl.rollout_size
            metadata['training_seconds'] = perf_counter() - started
            if metadata['status'] == 'running':
                metadata['status'] = 'interrupted'
            config_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return (ConstructionPolicy(result.best_policy, getattr(cost, 'base', cost), limits, native, rl, cost_function),
            pd.read_csv(output_dir / 'training.csv'), pd.read_csv(output_dir / 'validation.csv'))

def load_gat(output_dir):
    output_dir = Path(output_dir)
    metadata = json.loads((output_dir / 'config.json').read_text(encoding='utf-8'))
    native = OmegaConf.create(metadata['native'])
    policy, cost, limits = components(native, ProblemConfig(**metadata['problem']))
    load_model_weights(policy, output_dir / 'best.pt', map_location=DEVICE)
    ppo = PPOConfig.from_checkpoint(metadata.get('ppo', metadata.get('rl', {})))
    return ConstructionPolicy(policy, cost, limits, native, ppo, load_cost_function(output_dir))

def load_pretrained_gat(path, cfg, rl=None, *, cost_function=network_cost):
    """Load a standalone construction checkpoint with strict tensor matching."""
    rl = PPOConfig() if rl is None else rl
    payload = torch.load(Path(path), map_location=DEVICE, weights_only=True)
    state = payload.get('state_dict', payload)
    native = construction_config(cfg, rl)
    saved = payload.get('config', {})
    if 'model' in saved:
        native.model = OmegaConf.create(saved['model'])
    elif state['backbone_net.layers.0.lin_edge.weight'].shape[1] == 14:
        # The original raw artifact has ATT/WMC/RTT input slots. Keep all
        # columns and weights; the shared ATT/RTT objective supplies WMC=0.
        native.model.features.name = 'construction_v1'
        for group in ('pair', 'global_features', 'critic'):
            names = list(native.model.features[group])
            names.insert(names.index('pref.ATT') + 1, 'pref.WMC')
            native.model.features[group] = names
        native.model.route_generator.kwargs.feature_norm_unbiased = False
    policy, cost, limits = components(native, cfg)
    load_model_weights(policy, path, strict=True, map_location=DEVICE)
    return ConstructionPolicy(policy, cost, limits, native, rl, cost_function)

def train_or_load(config, datasets, cost_function=network_cost):
    output = Path(config.output_dir)
    model_dir = output / 'gat'
    check_output_writable(output)
    if config.train and config.pretrained_weights is None:
        check_training_destination(model_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_datasets(datasets, output / 'datasets.npz')
    (output / 'settings.json').write_text(json.dumps(asdict(config), indent=2), encoding='utf-8')
    histories, validations = {}, {}
    if config.pretrained_weights is not None:
        policy = load_pretrained_gat(config.pretrained_weights, config.problem, config.ppo,
                                     cost_function=cost_function)
        action, checkpoint = 'loaded', str(config.pretrained_weights)
    elif config.train:
        policy, history, validation = train_gat(
            datasets['train'], datasets['validation'], config.problem, config.ppo, model_dir,
            tensorboard_dir=output / 'tensorboard', cost_function=cost_function)
        histories['gat'], validations['gat'] = history, validation
        action, checkpoint = 'trained', str(model_dir)
    else:
        policy = load_gat(model_dir)
        histories['gat'] = pd.read_csv(model_dir / 'training.csv')
        validations['gat'] = pd.read_csv(model_dir / 'validation.csv')
        action, checkpoint = 'loaded', str(model_dir)
    summary = pd.DataFrame([dict(model='gat', training_action=action, checkpoint=checkpoint)]).set_index('model')
    return dict(policies={'gat': policy}, histories=histories, validations=validations,
                summary=summary, cost_function=cost_function)
