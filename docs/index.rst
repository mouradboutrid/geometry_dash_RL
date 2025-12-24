================================================================================
Geometry Dash Reinforcement Learning Agent
================================================================================

.. image:: https://img.shields.io/badge/Python-3.10+-blue.svg
    :target: https://www.python.org/downloads/
.. image:: https://img.shields.io/badge/License-MIT-green.svg
.. image:: https://img.shields.io/badge/Status-Active-brightgreen.svg

A sophisticated deep reinforcement learning system that masters Geometry Dash through curriculum-based training with expertise specialization. The project integrates native C++ game hooking via the Geode modding framework with a PyTorch-based training pipeline.

**Quick Links:**

- GitHub: https://github.com/yourusername/geometry_dash_RL
- Paper: [Available upon request]
- Discord: [Join our community]

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   introduction
   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Core Concepts
   :hidden:

   architecture
   algorithm
   curriculum

.. toctree::
   :maxdepth: 2
   :caption: Implementation Details
   :hidden:

   state_design
   reward_shaping
   geode_integration

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics
   :hidden:

   performance
   limitations
   future_work

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/agents
   api/environment
   api/curriculum

.. toctree::
   :maxdepth: 2
   :caption: Development
   :hidden:

   contributing
   changelog


Overview
--------

This project addresses the challenge of training a reinforcement learning agent to autonomously master a complex, precision-dependent action video game. Geometry Dash requires:

- **Precise timing** (frame-perfect inputs)
- **Long-horizon planning** (sequences of 100+ actions)
- **Mode switching** (physics changes mid-level)
- **Sparse rewards** (reward only on level completion)

Our solution combines:

✓ **Dueling Double Q-Learning**: Reduces overestimation bias in discrete action spaces
✓ **Curriculum Learning**: Progressively harder subsections with dense rewards
✓ **Cross-Language Synchronization**: Windows named pipes + spinlocks for C++/Python IPC
✓ **Expert Caching**: Reuses learned knowledge across curriculum slices
✓ **Mode-Aware Policy Transfer**: Specializes networks for Cube vs Ship physics

Key Features
------------

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: :octicon:`workflow` Automated Curriculum
      :link: curriculum
      :link-type: doc

      Progressively decompose levels into 8 manageable slices with automatic advancement criteria (70% success rate, 20-episode minimum).

   .. grid-item-card:: :octicon:`rocket` High Sample Efficiency
      :link: algorithm
      :link-type: doc

      Achieve >95% action optimality with ~500K total transitions through Double DQN and experience replay.

   .. grid-item-card:: :octicon:`cpu` Real-Time Integration
      :link: geode_integration
      :link-type: doc

      Synchronize with game engine at 60 FPS via low-latency shared memory and spinlock protocols.

   .. grid-item-card:: :octicon:`layers` Modular Architecture
      :link: architecture
      :link-type: doc

      Cleanly separated Python training pipeline, C++ game hooking, and expert reward shaping for extensibility.

Performance Highlights
----------------------

| Component | Metric | Value |
|-----------|--------|-------|
| Algorithm | Sample Efficiency | 95%+ action optimality |
| Training | Time to Completion | 18-24 hours (RTX 3080) |
| Inference | Network Latency | 3-5 ms per decision |
| Generalization | Success Rate | 71-95% per slice |
| Memory | Replay Buffer | 50K transitions |

Project Status
--------------

- Full training pipeline for Stereo Madness
- C++ Geode mod with real-time synchronization
- Curriculum learning with 8 slices
- Dual expert systems (Cube & Ship modes)
- Multi-level extension in progress
- Imitation learning bootstrapping planned

Requirements
------------

- **Hardware**: GPU (NVIDIA CUDA) or CPU
- **OS**: Windows 10/11
- **Python**: 3.10+
- **Geometry Dash**: v2.2074 with Geode loader

For installation instructions, see :doc:`installation`.

Example Usage
-------------

.. code-block:: python

   from Stereo_Madness.main import GDAgentOrchestrator
   
   # Initialize agent and environment
   orchestrator = GDAgentOrchestrator()
   
   # Start training loop
   orchestrator.train()
   
   # Monitor progress via curriculum manager
   current_slice = orchestrator.manager.get_current_slice()
   win_rate = orchestrator.manager.update(won_episode, steps_taken)

For detailed examples, see :doc:`quickstart`.

Contributing
------------

Contributions are welcome! Please see :doc:`contributing` for guidelines.

.. note::

   This project is actively maintained. For bug reports, feature requests, or questions, please open an issue on GitHub.

License
-------

MIT License. See LICENSE file for details.

Cite This Work
--------------

If you use this project in research, please cite:

.. code-block:: bibtex

   @software{gd_rl_2025,
     title={Geometry Dash Reinforcement Learning Agent},
     author={Boutrid, Mourad},
     year={2025},
     url={https://github.com/yourusername/geometry_dash_RL}
   }
