Contributing Guide
==================

How to contribute to this project.

Getting Started
---------------

1. **Fork the repository** on GitHub
2. **Clone locally**:

   .. code-block:: bash

      git clone https://github.com/yourusername/geometry_dash_RL.git
      cd geometry_dash_RL

3. **Create feature branch**:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

4. **Install dev dependencies**:

   .. code-block:: bash

      pip install -r docs/requirements.txt
      pip install pytest pytest-cov black mypy

Development Workflow
--------------------

**Before Coding**

1. Check existing issues/PRs (avoid duplicates)
2. Open issue describing proposed change
3. Discuss design before implementation

**Code Style**

- **Python**: PEP 8 (use black formatter)
- **C++**: Google C++ style guide
- **Docstrings**: Google-style (see existing code)

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

.. code-block:: text

   Examples:
   [FEAT] Add meta-learning training loop
   [FIX] Correct spinlock timeout in MemoryBridge
   [DOCS] Update algorithm section with references
   [TEST] Add unit tests for ReplayBuffer
   [REFACTOR] Simplify CurriculumManager initialization

Contribution Areas
-------------------

**High Priority** (18-24 month roadmap)

1. **Multi-Level Support** (3-4 weeks)
   - Design level-agnostic architecture
   - Test on 2-3 additional levels
   - Automated slice discovery (longer term)

2. **Domain Adaptation** (4-6 weeks)
   - Transfer learning framework
   - Validate on new levels
   - Benchmark speedup

3. **Model Compression** (2-3 weeks)
   - Pruning pipeline
   - Quantization
   - Latency benchmarking

4. **Distributed Training** (6-8 weeks)
   - Multi-GPU support
   - Asynchronous learning
   - Fault tolerance

**Medium Priority** (6-12 month roadmap)

5. **Inverse RL** (8-12 weeks)
   - Collect human demonstrations
   - Implement inverse RL algorithm
   - Validate reward learning

6. **Meta-Learning** (12-16 weeks)
   - Implement MAML or similar
   - Meta-train on multiple levels
   - Few-shot adaptation

7. **Imitation Learning** (6-8 weeks)
   - Behavioral cloning
   - DAgger (active learning)
   - Demonstration collection UI

**Lower Priority** (12+ month roadmap)

8. **Web-based Version**
9. **Docker Container**
10. **Multi-version GD Support**

Pick an area, open an issue, and we'll discuss!

Pull Request Process
--------------------

1. **Ensure code quality**:

   .. code-block:: bash

      black Stereo_Madness/
      mypy Stereo_Madness/
      pytest tests/

2. **Update documentation**:
   - Docstrings for new functions
   - README changes if user-facing
   - Architecture docs if structural changes

3. **Open PR with description**:

   .. code-block:: markdown

      ## Description
      Briefly describe what this PR does.

      ## Motivation
      Why is this change needed?

      ## Changes
      - Change 1
      - Change 2

      ## Testing
      How was this tested?

      ## Related Issues
      Closes #123

4. **Respond to feedback** (within 48 hours)

5. **Merge** (after approval)

Code Review Guidelines
----------------------

**For Authors**

- Keep PRs focused (one feature per PR)
- Aim for <500 lines per PR (break up large changes)
- Explain non-obvious decisions
- Link related issues

**For Reviewers**

- Check correctness (logic, edge cases)
- Verify style compliance
- Test locally if possible
- Be constructive and collaborative

Documentation
--------------

This documentation is built with Sphinx and hosted on Read the Docs.

**To modify docs**:

1. Edit ``.rst`` files in ``docs/``
2. Build locally:

   .. code-block:: bash

      cd docs/
      make html
      open _build/html/index.html

3. PR with documentation changes

**Adding new pages**:

1. Create ``docs/section_name.rst``
2. Add to ``docs/index.rst`` toctree
3. Follow existing structure and style

Reporting Issues
----------------

**Security Issues**

Do NOT open public issue. Email maintainer privately.

**Bugs**

.. code-block:: markdown

   ## Description
   Describe the bug clearly.

   ## Steps to Reproduce
   1. Step 1
   2. Step 2
   3. ...

   ## Expected Behavior
   What should happen?

   ## Actual Behavior
   What actually happened?

   ## Environment
   - Python version: 3.10
   - PyTorch version: 2.0
   - GPU: RTX 3080
   - Windows version: 10

   ## Logs
   Paste relevant error messages or logs.

**Feature Requests**

.. code-block:: markdown

   ## Feature Description
   What feature would you like?

   ## Motivation
   Why is this useful?

   ## Proposed Solution
   How might we implement it?

   ## Alternatives
   Other approaches?

Community Guidelines
--------------------

- Be respectful and inclusive
- Assume good intentions
- Welcome diverse perspectives
- Focus on ideas, not people

Project Maintainers
-------------------

- **Lead**: Mourad Boutrid (@username)
- **Reviewers**: TBD

Contact for questions: email, GitHub discussions, or Discord.

License
-------

By contributing, you agree your code is licensed under MIT (same as project).

Recognition
-----------

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Celebrated in community! 🎉

Code of Conduct
---------------

This project adheres to the Contributor Covenant Code of Conduct.

Expected behavior:
- Professional interactions
- Welcoming to all backgrounds
- Respect differences

Unacceptable behavior:
- Harassment or discrimination
- Threats or violence
- Unwelcome advances

Violations reported to maintainers; may result in ban.

See CODE_OF_CONDUCT.md for full details.

Questions?
----------

Open a GitHub discussion or send email. Welcome aboard! 👋
