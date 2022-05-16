from datetime import datetime
from typing import Any, Optional

import torch.nn.functional as F
from omegaconf import ListConfig
from pydantic import BaseModel
from rewards_csv import ContinuousRewardsCsv, SimulationRewardsCsv
from sentient_util.pydantic_config import (
    DataframeEnvConfig,
    SimEnv0_Config,
    SimEnv1_Config,
)


class SimTraderConfig(BaseModel):
    trader: str
    QL: float
    depth: float
    allowOrderConflicts: bool
    feeRate: float
    precision: int
    pdf: str


class NoNoise(BaseModel):
    sigma: float
    type: str


class Network(BaseModel):
    activations: ListConfig | list[str]
    hidden_dims: ListConfig | list[int]
    conv_filters: ListConfig | list[int] = [64, 32]
    kernel_size: int = 2

    class Config:
        arbitrary_types_allowed = True


class EpsilonDecay(BaseModel):
    type: str  # LINEAR or EXPONENTIAL
    rate: float  # Explore decay rate
    end: float  # Final (steady-state) explore probability
    start: float  # Initial explore probability


class TorchModels(BaseModel):
    Actor: Network
    Critic: Network


class Misc(BaseModel):
    csv_path: Optional[str] = None
    generate_csv: bool = False
    leave_progress_bar: bool = True
    log_interval: int = 1000
    log_level: str = "INFO"
    market: Optional[str] = None
    seed: int = 7
    record_state = False
    csv_class: str = "ContinuousRewardsCsv"


class BaseAgent(BaseModel):
    actor_lr: float
    config_class: str = "DefaultProcessConfig"
    critic_lr: float
    episodes: int
    epsilon_decay: Optional[EpsilonDecay] = None
    exploration: int = 100
    gamma: float
    gradient_clipping: Any
    # max_norm:                        1.0
    # norm_type:                       2.0
    network: Optional[str] = None
    purge_network: bool = True
    target_update_interval: int = 2
    training: bool = True
    training_interval: int
    loss: Any = F.smooth_l1_loss
    save_onnx: bool = False


class DdpgAgent(BaseAgent):
    agent: str = "ddpg.ddpg_agent"
    action_noise: Optional[NoNoise] = None
    batch_size: int
    buffer_size: int
    torch_models: TorchModels
    tau: float = 0.001


class DdpgConv1dAgent(BaseAgent):
    agent: str = "ddpg.ddpg_agent_conv1d"
    action_noise: Optional[NoNoise] = None
    batch_size: int
    buffer_size: int
    torch_models: TorchModels
    tau: float = 0.001

    class Config:
        arbitrary_types_allowed = True


class Td3Agent(BaseAgent):
    agent: str = "td3.td3_agent"
    action_noise: Optional[NoNoise] = None
    batch_size: int
    buffer_size: int
    noise_clip: float
    policy_noise: float
    torch_models: TorchModels
    tau: float = 0.001


class Td3Conv1dAgent(BaseAgent):
    agent: str = "td3.td3_agent_conv1d"
    action_noise: Optional[NoNoise] = None
    batch_size: int
    buffer_size: int
    torch_models: TorchModels
    tau: float = 0.001
    policy_noise: float
    noise_clip: float

    class Config:
        arbitrary_types_allowed = True


class SacConv1dAgent(BaseAgent):
    agent: str = "sac.sac_agent_conv1d"
    action_noise: Optional[NoNoise] = None
    batch_size: int
    buffer_size: int
    torch_models: TorchModels
    tau: float = 0.001
    alpha: float = 0.2
    automatic_entropy_tuning: bool = True

    class Config:
        arbitrary_types_allowed = True


class SacAgent(BaseAgent):
    agent: str = "sac.sac_agent"
    action_noise: Optional[NoNoise] = None
    batch_size: int
    buffer_size: int
    torch_models: TorchModels
    tau: float = 0.001
    alpha: float = 0.2
    automatic_entropy_tuning: bool = True


class PpoAgent(BaseAgent):
    agent: str = "ppo.ppo_agent"
    torch_models: TorchModels | None
    K_epochs: int
    eps_clip: float
    action_std_decay_freq: float
    action_std_init: float
    # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    action_std_decay_rate: float
    # minimum action_std (stop decay after action_std <= min_action_std)
    min_action_std: float
    batch_size: int = 0
    buffer_size: int = 0


class DefaultAgent(BaseAgent):
    agent: str = "ppo.ppo_agent"
    action_noise: Optional[NoNoise] = None
    batch_size: int
    buffer_size: int
    torch_models: TorchModels
    tau: float = 0.001


class SimAgent(BaseModel):
    agent: str = "simulator.simulator_agent"
    trader_config: SimTraderConfig
    episodes: int


class ChildAgent(BaseModel):
    batch_size: int | None
    buffer_size: int | None
    episodes: int | None


class ChildProcess(BaseModel):
    launch_delay: int
    launch_interval: int
    agent: ChildAgent | None
    misc: Misc
    env_config: Any


class ProcessConfig(BaseModel):
    agent: BaseAgent
    env_config: Any
    misc: Misc
    child_process: ChildProcess | None
    console: str = "logging"
    instance_id: int = 0


class DdpgProcessConfig(BaseModel):
    agent: DdpgAgent
    env_config: DataframeEnvConfig
    misc: Misc
    child_process: ChildProcess | None
    console: str = "logging"
    instance_id: int = 0


class Ddpg_SimEnv0_Config(BaseModel):
    agent: DdpgAgent
    env_config: SimEnv0_Config
    misc: Misc
    child_process: ChildProcess | None
    console: str = "logging"
    instance_id: int = 0


class DDPG_SimEnv1_Config(BaseModel):
    agent: DdpgConv1dAgent
    env_config: SimEnv1_Config
    misc: Misc
    child_process: ChildProcess | None = None
    console: str = "logging"
    instance_id: int = 0


class Td3ProcessConfig(BaseModel):
    agent: Td3Agent
    env_config: DataframeEnvConfig
    misc: Misc
    child_process: ChildProcess | None
    console: str = "logging"
    instance_id: int = 0


class TD3_SimEnv1_Config(BaseModel):
    agent: Td3Conv1dAgent
    env_config: SimEnv1_Config
    misc: Misc
    child_process: ChildProcess | None = None
    console: str = "logging"
    instance_id: int = 0


class SacProcessConfig(BaseModel):
    agent: SacAgent
    env_config: DataframeEnvConfig
    misc: Misc
    child_process: ChildProcess | None
    console: str = "logging"
    instance_id: int = 0


class SAC_SimEnv1_Config(BaseModel):
    agent: SacConv1dAgent
    env_config: SimEnv1_Config
    misc: Misc
    child_process: ChildProcess | None = None
    console: str = "logging"
    instance_id: int = 0


class PpoProcessConfig(BaseModel):
    agent: PpoAgent
    env_config: DataframeEnvConfig | SimEnv0_Config
    misc: Misc
    child_process: ChildProcess | None
    console: str = "logging"
    instance_id: int = 0


class SimProcessConfig(BaseModel):
    agent: SimAgent
    env_config: SimEnv0_Config
    misc: Misc
    console: str = "logging"
    instance_id: int = 0


class DefaultProcessConfig(BaseModel):
    agent: DefaultAgent
    env_config: DataframeEnvConfig | SimEnv0_Config
    misc: Misc
    console: str = "logging"
    instance_id: int = 0


class PytestModel(BaseModel):
    misc: Misc
    console: str = "logging"
    instance_id: int = 0
    env_name: str


class DdpgPytestConfig(PytestModel):
    agent: DdpgAgent


class Td3PytestConfig(PytestModel):
    agent: Td3Agent


class SacPytestConfig(PytestModel):
    agent: SacAgent
