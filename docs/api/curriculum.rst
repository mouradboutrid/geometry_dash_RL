API Reference - Curriculum
===========================

Documentation for curriculum learning components.

.. module:: curriculum

CurriculumManager
-----------------

.. automodule:: curriculum.manager
   :members:
   :undoc-members:

.. py:class:: CurriculumManager(curriculum_file=None)

   Manages progressive curriculum learning progression across **9 strategic slices**. 
   It tracks the agent's performance and handles the automated transition between level segments.

   **Tracked Metrics:**
   * Current slice index (0-8)
   * Episode count per slice
   * Rolling success rate (50-episode window)
   * Best success rate per slice
   * Total training steps across all sessions

   **File Format:**
   The state is persisted as a JSON file at ``CURRICULUM_FILE``:

   .. code-block:: json

      {
        "slices": [
          {
            "id": 1,
            "description": "Intro - Simple Cube jumps",
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

   .. py:method:: get_current_slice()

      Get the definition of the slice currently being trained.

      :return: Slice dict with keys: ``id``, ``start``, ``end``, ``mode``, ``target_w``, ``description``
      :rtype: dict

   .. py:method:: record_episode(success, reward, steps)

      Record outcome and evaluate slice advancement.

      

      :param bool success: True if agent reached the ``end`` percentage of the slice
      :param float reward: Total reward received in the episode
      :param int steps: Steps taken in the episode
      :return: Tuple of (advanced, new_slice)
      :rtype: tuple[bool, dict or None]

      **Advancement Criteria:**
      * **Volume**: $\ge 20$ episodes in current slice.
      * **Consistency**: $\ge 70\%$ rolling success rate (last 50 episodes).

   .. py:method:: get_progress()

      :return: Overall progress calculated as ``current_slice / 9.0``
      :rtype: float

   .. py:method:: get_expert_for_relay(source_slice=None)

      Get checkpoint path for a previous slice expert.

      :param int source_slice: Index of the expert to load (default: current - 1)
      :return: Path to checkpoint file or None
      :rtype: str or None

   .. py:method:: is_complete()

      :return: True if the final slice (ID 9) has met advancement criteria
      :rtype: bool

   .. py:method:: save()

      Commits the current curriculum state to the disk.

   .. py:method:: get_stats()

      :return: Dict with 'current_slice', 'episodes_total', 'hours_trained', 'avg_reward'
      :rtype: dict

Slice Definitions
-----------------

The curriculum is decomposed into 9 specific segments:

.. list-table:: Level Segmentation
   :header-rows: 1
   :widths: 5 35 15 15 15 15

   * - ID
     - Description
     - Start%
     - End%
     - Target%
     - Mode
   * - 1
     - Intro - Simple Cube jumps
     - 0.0
     - 10.0
     - 10.0
     - Cube (0)
   * - 2
     - Pre-Triple Spike - Standard Cube
     - 10.0
     - 20.0
     - 20.0
     - Cube (0)
   * - 3
     - The Triple Spike - High Precision
     - 20.0
     - 31.0
     - 30.0
     - Cube (0)
   * - 4
     - First Ship Section - Stability
     - 30.0
     - 47.0
     - 47.0
     - Ship (1)
   * - 5
     - Mid Level Cube - Rhythm
     - 47.0
     - 58.0
     - 58.0
     - Cube (0)
   * - 6
     - Mid Level Cube II - Transitions
     - 58.0
     - 68.0
     - 68.0
     - Cube (0)
   * - 7
     - Pre-Final Cube
     - 68.0
     - 77.0
     - 77.0
     - Cube (0)
   * - 8
     - Clutterfunk - Complex jumps
     - 77.0
     - 86.0
     - 86.0
     - Cube (0)
   * - 9
     - Final Ship - Victory Lap
     - 86.0
     - 97.0
     - 97.0
     - Ship (1)

Physics Modes
-------------

Each slice specifies required physics mode(s):

- **Cube (mode=0)**: Jump-based (binary action controls hold-to-charge).
- **Ship (mode=1)**: Velocity-based (binary action controls vertical thrust).

Mode-Specific Expert Caching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When transitioning modes, fresh networks are initialized to avoid gradient interference:

.. code-block:: python

   curr = CurriculumManager()
   current = curr.get_current_slice()

   # Example: Switching to Slice 4 (Ship)
   if current['mode'] == 1:
       # Load Slice 3 (Cube) expert to handle the first 30% of the level
       relay_expert = curr.get_expert_for_relay(source_slice=2)
       agent.load(relay_expert)

       # Reset network for the new Ship physics
       agent.reset_network()

Example Usage
-------------

Training Loop Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from curriculum.manager import CurriculumManager
   from agents.ddqn import Agent

   curr = CurriculumManager()
   agent = Agent(INPUT_DIM, OUTPUT_DIM)

   while not curr.is_complete():
       slice_data = curr.get_current_slice()
       obs, _ = env.reset()
       
       # ... Training Loop ...
       
       advanced, new_slice = curr.record_episode(success, reward, steps)

       if advanced:
           print(f"Advanced to {new_slice['description']}!")
           curr.save()
           
           # Load expert for relay race (handling the segment before the current slice)
           expert_path = curr.get_expert_for_relay()
           if expert_path:
               agent.load(expert_path)

Manual Curriculum Control
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   curr = CurriculumManager()

   # Jump to Slice 4
   curr.current_slice = 3
   curr.save()

   # Reset training history for a specific slice
   curr.slices[3]['episodes_trained'] = 0
   curr.save()

Progression Criteria
--------------------

Automatic advancement requires:

1. **Episode Count**: $\ge 20$ episodes in current slice.
2. **Success Rate**: $\ge 70\%$ over rolling 50-episode window.

Historical Advancement Rates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on empirical training sessions:

- **Slices 1-3 (Cube Intro/Spikes)**: ~150 episodes total (5-6 hours).
- **Slice 4 (First Ship)**: ~60 episodes (2-3 hours).
- **Slices 5-8 (Mid/Clutterfunk Cube)**: ~140 episodes (6-8 hours).
- **Slice 9 (Final Ship)**: ~50 episodes (2 hours).

**Total Progression**: ~400 episodes (~18-26 hours on RTX 3080).

Troubleshooting
---------------

**"Advanced" but agent still struggling**
The agent may have had a "lucky" run. Manually revert:
.. code-block:: python

   curr.current_slice -= 1
   curr.save()

**Want deterministic curriculum**
Set ``requires_percent`` to 100 in the config and modify ``record_episode()`` to check for a fixed episode count instead of success rate.
