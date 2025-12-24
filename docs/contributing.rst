Contributing Guide
==================

How to contribute to this project.

Getting Started
---------------

1. **Fork the repository** on GitHub
2. **Clone locally**:

   .. code-block:: bash

      git clone https://github.com/mouradboutrid/geometry_dash_RL.git
      cd geometry_dash_RL

3. **Create feature branch**:

   .. code-block:: bash

      git checkout -b feature/feature-name

4. **Install dev dependencies**:

   .. code-block:: bash

      pip install -r docs/requirements.txt
      pip install pytest pytest-cov black mypy

Development Workflow
--------------------

.. code-block:: bash

   # Format Python code
   black Stereo_Madness/

   # Type check
   mypy Stereo_Madness/

**Testing**

.. code-block:: bash

   # Run tests (when available)
   pytest tests/

   # Code coverage
   pytest --cov=Stereo_Madness tests/

**Commit Messages**

Format: ``[CATEGORY] Brief description``


Contribution Areas
-------------------

**High Priority** 

1. **Multi-Level Support** 
   - Design level-agnostic architecture
   - Test on 2-3 additional levels
   - Automated slice discovery (longer term)

2. **Domain Adaptation**
   - Transfer learning framework
   - Validate on new levels
   - Benchmark speedup

3. **Model Compression**
   - Pruning pipeline
   - Quantization
   - Latency benchmarking

4. **Distributed Training** 
   - Multi-GPU support
   - Asynchronous learning
   - Fault tolerance

**Medium Priority**

5. **Inverse RL** 
   - Collect human demonstrations
   - Implement inverse RL algorithm
   - Validate reward learning

6. **Meta-Learning** 
   - Implement MAML or similar
   - Meta-train on multiple levels
   - Few-shot adaptation

7. **Imitation Learning** 
   - Behavioral cloning
   - DAgger (active learning)
   - Demonstration collection UI

**Lower Priority** 

8. **Web-based Version**
9. **Docker Container**
10. **Multi-version GD Support**

Questions?
----------

Open a GitHub discussion or send email. Welcome aboard! 
