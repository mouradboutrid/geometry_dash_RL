API Reference - Curriculum
===========================

Documentation for curriculum learning components.

.. module:: curriculum

CurriculumManager
-----------------

.. automodule:: curriculum.manager
   :members:
   :undoc-members:

.. py:class:: CurriculumManager

   Manages progressive curriculum learning progression across 8 slices.

   Tracks:
   - Current slice index (0-7)
   - Episode count per slice
   - Rolling success rate (50-episode window)
   - Best success rate per slice
   - Total training steps

   File Format:

   The curriculum state is saved as JSON at CURRICULUM_FILE:

   .. code-block:: json

      {
        "slices": [
          {
            "name": "Slice 01: First Spike (0-15%)",
            "best_success_rate": 0.82,
            "episodes_trained": 125,
            "best_model_path": "checkpoints/slice_01_best.pth"
          },
          ...
        ],
        "current_slice": 2,
        "total_steps": 45820,
        "created_at": "2024-01-15T10:22:33",
        "last_updated": "2024-01-20T14:55:22"
      }

   .. py:method:: __init__(curriculum_file=None)

      Load or create new curriculum state.

      :param str curriculum_file: Path to curriculum JSON (default: CURRICULUM_FILE)

   .. py:method:: get_current_slice() -> dict

      Get current slice definition.

      :return: Slice dict with 'name', 'start_percent', 'end_percent', 'mode'
      :rtype: dict

      Example:

      .. code-block:: python

         curr = CurriculumManager()
         current = curr.get_current_slice()
         print(current['name'])  # "Slice 01: First Spike (0-15%)"
         print(current['mode'])   # 0 (cube) or 1 (ship)

   .. py:method:: record_episode(success, reward, steps)

      Record episode outcome and check for slice advancement.

      :param bool success: Episode succeeded (reached end_percent)
      :param float reward: Total episode reward
      :param int steps: Steps taken in episode
      :return: Tuple of (advanced, new_slice) - advanced=True if moved to new slice
      :rtype: tuple[bool, dict or None]

      Advancement Criteria:
      - Requires ≥20 episodes in current slice
      - Requires ≥70% rolling success rate (50-episode window)

      Example:

      .. code-block:: python

         curr = CurriculumManager()
         for episode in range(1000):
             success = (final_percent >= current_slice['end_percent'])
             advanced, new_slice = curr.record_episode(success, total_reward, steps)

             if advanced:
                 print(f"Advanced to {new_slice['name']}!")
                 # Load expert for relay race
                 agent.load(new_slice['expert_cache'])

   .. py:method:: get_progress() -> float

      Get overall curriculum progress (0.0 to 1.0).

      :return: Slice index / 8
      :rtype: float

   .. py:method:: get_expert_for_relay(source_slice=None) -> str or None

      Get checkpoint path for relay race expert.

      :param int source_slice: Source slice (default: current - 1)
      :return: Path to expert checkpoint or None
      :rtype: str or None

      Example:

      .. code-block:: python

         # Start new slice training
         expert_path = curr.get_expert_for_relay()
         if expert_path:
             agent.load(expert_path)
             print("Loaded relay expert")

   .. py:method:: is_complete() -> bool

      Check if all 8 slices completed.

      :return: True if current_slice >= 7 and last slice complete
      :rtype: bool

   .. py:method:: save()

      Write curriculum state to disk.

   .. py:method:: get_stats() -> dict

      Get human-readable curriculum statistics.

      :return: Dict with 'current_slice', 'episodes_total', 'hours_trained', 'avg_reward'
      :rtype: dict

Slice Definitions
-----------------

.. automodule:: curriculum.manager
   :members: SLICES

The curriculum is decomposed into 8 slices:

.. list-table:: Curriculum Slices
   :header-rows: 1
   :widths: 10 30 20 20 20

   * - Slice
     - Description
     - Start%
     - End%
     - Mode
   * - 1
     - First Spike (obstacle avoidance warmup)
     - 0%
     - 15%
     - Cube
   * - 2
     - Platform Sequence (jumping mechanics)
     - 15%
     - 35%
     - Cube
   * - 3
     - Wave + Spike (physics transition)
     - 35%
     - 55%
     - Ship+Cube
   * - 4
     - UFO Section (altitude control)
     - 55%
     - 70%
     - Ship
   * - 5
     - Complex Platforming (sustained jumping)
     - 70%
     - 80%
     - Cube
   * - 6
     - Wave Finale (multi-mode)
     - 80%
     - 90%
     - Ship+Cube
   * - 7
     - Dual Mode Mastery (fluent switching)
     - 90%
     - 99%
     - Both
   * - 8
     - Full Level (end-to-end)
     - 0%
     - 100%
     - Both

Physics Modes
-------------

Each slice specifies required physics mode(s):

- **Cube (mode=0)**: Jump-based (binary action controls hold-to-charge)
- **Ship (mode=1)**: Velocity-based (binary action controls vertical thrust)
- **Both**: Slice requires both modes (expects mode-switch transitions)

Mode-Specific Expert Caching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When transitioning between physics modes, fresh networks are initialized:

.. code-block:: python

   curr = CurriculumManager()
   current = curr.get_current_slice()

   # Start slice 4 (ship mode)
   if current['mode'] == 1:  # Ship
       # Load cube expert from slice 3 for relay race to 55%
       relay_expert = curr.get_expert_for_relay(source_slice=2)
       agent.load(relay_expert)

       # At 55%, switch to fresh ship network
       # (don't transfer cube weights to ship)
       agent.reset_network()
       print("Fresh ship network initialized")

See :doc:`../curriculum` for detailed explanation.

Example Usage
-------------

Training Loop Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from curriculum.manager import CurriculumManager
   from core.environment import GeometryDashEnv
   from agents.ddqn import Agent
   from config import *

   curr = CurriculumManager()
   env = GeometryDashEnv()
   agent = Agent(INPUT_DIM, OUTPUT_DIM)

   step = 0
   while not curr.is_complete():
       current_slice = curr.get_current_slice()
       end_percent = current_slice['end_percent']
       mode = current_slice['mode']

       obs, _ = env.reset()
       episode_reward = 0
       success = False

       for _ in range(50000):  # Max steps per episode
           action = agent.select_action(obs, is_training=True)
           obs, reward, done, _, info = env.step(action)
           episode_reward += reward
           step += 1

           if info['percent'] >= end_percent:
               success = True
               break
           if done:
               break

       # Record episode and check for advancement
       advanced, new_slice = curr.record_episode(success, episode_reward, step)

       if advanced:
           print(f"\n✓ Advanced to {new_slice['name']}!")
           curr.save()

           # Load expert for relay race
           expert_path = curr.get_expert_for_relay()
           if expert_path:
               agent.load(expert_path)

       print(f"Slice {curr.current_slice+1}/8 | Success: {success} | Reward: {episode_reward:.0f}")

Manual Curriculum Control
~~~~~~~~~~~~~~~~~~~~~~~~~~

To skip or reset curriculum:

.. code-block:: python

   curr = CurriculumManager()

   # Jump to slice 3
   curr.current_slice = 2
   curr.save()

   # Reset slice 2
   curr.slices[2]['best_success_rate'] = 0.0
   curr.slices[2]['episodes_trained'] = 0
   curr.save()

   # Regenerate from scratch
   curr2 = CurriculumManager('fresh_curriculum.json')
   curr2.save()

Progression Criteria
--------------------

Automatic advancement requires:

1. **Episode Count**: ≥20 episodes in current slice
2. **Success Rate**: ≥70% over rolling 50-episode window

Rationale:

- **70% threshold**: Empirically sufficient to master slice mechanics
- **50-episode window**: Balances consistency vs. training time
- **20 episode minimum**: Prevents lucky early advancement

Historical Advancement Rates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on empirical training:

- Slice 1-2 (cube warmup): ~120 episodes total (4-5 hours)
- Slice 3 (physics transition): ~180 episodes (6-7 hours) - hardest
- Slice 4-5 (secondary mode): ~90 episodes (3-4 hours)
- Slice 6-7 (multi-mode mastery): ~140 episodes (5-6 hours)
- Slice 8 (full level): ~190 episodes (7-8 hours) - hardest

**Total**: ~270 episodes (~18-26 hours on RTX 3080, ~1-2 weeks on CPU)

See :doc:`../performance` for actual training curves.

Troubleshooting
---------------

**"Advanced" but agent still struggling**

The agent may have advanced due to lucky streak. Resume training:

.. code-block:: python

   curr = CurriculumManager()
   curr.current_slice -= 1  # Go back one slice
   curr.save()
   # Retrain

**Progress stalled at slice X**

Check if reward coefficients are appropriate:

.. code-block:: python

   curr = CurriculumManager()
   print(curr.get_stats())
   # If avg_reward << expected, adjust config.py reward weights

**Want deterministic curriculum**

Set all slices to 100% in slice_definitions.json:

.. code-block:: json

   "requires_percent": 100

Then modify record_episode() check:

.. code-block:: python

   # Check success by fixed episode count instead
   success = (steps > min_steps)

See :doc:`../curriculum` for theoretical background.
