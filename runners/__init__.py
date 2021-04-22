REGISTRY = {}

from .episode_runner import EpisodeRunner
from .episode_runner_V2 import EpisodeRunnerV2

REGISTRY["episode"] = EpisodeRunner
REGISTRY["episode_V2"] = EpisodeRunnerV2