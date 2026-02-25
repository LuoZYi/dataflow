from __future__ import annotations

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from omegaconf import OmegaConf


class SAM3ExternalTrainer:
    """
    Launch SAM3's official training entry:
      python sam3/train/train.py -c <CONFIG> [--use-cluster ...] [--num-gpus ...] [--num-nodes ...]

    We optionally "patch" the SAM3 config so that paths.experiment_log_dir points to our out_dir,
    to keep everything under your unified runs/ directory.
    """

    def __init__(
        self,
        out_dir: str,
        sam3_repo_root: str,
        sam3_config: str,
        # launcher args (these are passed to SAM3 train.py)
        use_cluster: Optional[bool] = None,
        partition: Optional[str] = None,
        account: Optional[str] = None,
        qos: Optional[str] = None,
        num_gpus: Optional[int] = None,
        num_nodes: Optional[int] = None,
        # config patching
        patch_experiment_log_dir: bool = True,
        # extra env
        extra_env: Optional[Dict[str, str]] = None,
        # extra cli args appended as raw string (optional)
        extra_cli: str = "",
    ):
        self.out_dir = str(out_dir)
        self.sam3_repo_root = str(sam3_repo_root)
        self.sam3_config = str(sam3_config)

        self.use_cluster = use_cluster
        self.partition = partition
        self.account = account
        self.qos = qos
        self.num_gpus = num_gpus
        self.num_nodes = num_nodes

        self.patch_experiment_log_dir = bool(patch_experiment_log_dir)
        self.extra_env = dict(extra_env or {})
        self.extra_cli = str(extra_cli)

    def _resolve_config_path(self) -> Path:
        """
        sam3_config can be:
          - absolute path
          - relative to sam3_repo_root
        """
        p = Path(self.sam3_config)
        if p.is_absolute():
            return p
        return (Path(self.sam3_repo_root) / p).resolve()

    def _patched_config(self, src: Path) -> Path:
        """
        Make a patched copy of SAM3 job config under out_dir to force experiment_log_dir.
        """
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cfg = OmegaConf.load(str(src))

        # Make sure paths exists
        if "paths" not in cfg:
            cfg.paths = {}

        # Force all outputs to your out_dir
        cfg.paths.experiment_log_dir = str(out_dir)

        # Keep launcher.experiment_log_dir consistent if present
        if "launcher" in cfg:
            cfg.launcher.experiment_log_dir = "${paths.experiment_log_dir}"

        # Keep submitit.use_cluster consistent if present and user overrides
        if self.use_cluster is not None and "submitit" in cfg:
            cfg.submitit.use_cluster = bool(self.use_cluster)

        # Save patched config
        dst = out_dir / "sam3_job_patched.yaml"
        OmegaConf.save(cfg, str(dst))
        return dst

    def run(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # env
        env = os.environ.copy()
        # If you installed `pip install -e ".[train]"` under repo root, PYTHONPATH isn't required,
        # but adding it doesn't hurt and helps when debugging.
        env["PYTHONPATH"] = f"{self.sam3_repo_root}:" + env.get("PYTHONPATH", "")
        env.update(self.extra_env)

        src_cfg = self._resolve_config_path()
        if not src_cfg.exists():
            raise FileNotFoundError(f"SAM3 config not found: {src_cfg}")

        job_cfg = self._patched_config(src_cfg) if self.patch_experiment_log_dir else src_cfg

        # command (match README)
        cmd = [
            "python",
            "-u",
            "sam3/train/train.py",
            "-c",
            str(job_cfg),
        ]

        if self.use_cluster is not None:
            cmd += ["--use-cluster", "1" if self.use_cluster else "0"]
        if self.partition:
            cmd += ["--partition", str(self.partition)]
        if self.account:
            cmd += ["--account", str(self.account)]
        if self.qos:
            cmd += ["--qos", str(self.qos)]
        if self.num_gpus is not None:
            cmd += ["--num-gpus", str(int(self.num_gpus))]
        if self.num_nodes is not None:
            cmd += ["--num-nodes", str(int(self.num_nodes))]

        if self.extra_cli.strip():
            # allow raw string append
            cmd += self.extra_cli.strip().split()

        # record command
        (out_dir / "sam3_cmd.sh").write_text(" ".join(cmd) + "\n", encoding="utf-8")

        logging.info(f"[sam3_external] cwd={self.sam3_repo_root}")
        logging.info(f"[sam3_external] job_cfg={job_cfg}")
        logging.info(f"[sam3_external] cmd={' '.join(cmd)}")

        p = subprocess.Popen(cmd, cwd=self.sam3_repo_root, env=env)
        ret = p.wait()
        if ret != 0:
            raise RuntimeError(f"SAM3 failed (exit={ret}). See logs under: {out_dir}")

        logging.info("[sam3_external] DONE")