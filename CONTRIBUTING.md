# Bulldozer Contributing guide

1. [Report issues](#report-issues)
2. [Contributing workflow](#contributing-workflow)
3. [Coding guide](#coding-guide)
4. [Merge request acceptation process](#merge-request-acceptation-process)

**Contributions are welcome and greatly appreciated!**

# Report issues

Any proven or suspected malfunction should be traced in a bug report, the latter being an issue in the **Bulldozer** [github repository](https://github.com/CNES/bulldozer).

**Don't hesitate to do so: It is best to open a bug report and quickly resolve it than to let a problem remains in the project.**
**Notifying the potential bugs is the first way for contributing to a software.**



In the problem description, be as accurate as possible. Include:
* The procedure used to initialize the environment
* The incriminated command line or python function
* The content of the output log file

# Contributing workflow

Any code modification requires a Merge Request. It is forbidden to push patches directly into master (this branch is protected).

It is recommended to open your Merge Request as soon as possible in order to inform the developers of your ongoing work.
Please add `WIP:` before your Merge Request title if your work is in progress: This prevents an accidental merge and informs the other developers of the unfinished state of your work.

The Merge Request shall have a short description of the proposed changes. If it is relative to an issue, you can signal it by adding `Closes xx` where xx is the reference number of the issue.

Likewise, if you work on a branch (which is recommended), prefix the branch's name by `xx-` in order to link it to the xx issue.

Classical workflow is :
* Create an issue (or begin from an existing one)
* Create a Merge Request from the issue: a MR is created accordingly with "WIP:", "Closes xx" and associated "xx-name-issue" branch
* Hack code from a local working directory
* If you use Cython, you must name your C++ files with the following format: `c_<filename>.[cpp/h]`
* Git add, commit and push from local working clone directory
* Follow [Conventional commits](https://www.conventionalcommits.org/) specifications for commit messages
* Launch the tests on your modifications
* When finished, change your Merge Request name (erase "WIP:" in title) and ask to review the code (see below[Merge request acceptation process](#merge-request-acceptation-process))

# Coding guide

Here are some rules to apply when developing a new functionality:
* Include a comments ratio high enough and use explicit variables names. A comment by code block of several lines is necessary to explain a new functionality.
* The usage of the `print()` function is forbidden: use the **Bulldozer** internal logger instead.
* Each new functionality shall have a corresponding test in its module's test file. This test shall, if possible, check the function's outputs and the corresponding degraded cases.
* All functions shall be documented (object, parameters, return values).
* If major modifications of the user interface or of the tool's behaviour are done, update the user documentation (and the notebooks if necessary).
* Do not add new dependencies unless it is absolutely necessary, and only if it has a permissive license.
* Use the type hints provided by the `typing` python module.

# Merge request acceptation process

The Merge Request will be merged into master after being reviewed by **Bullodzer** steering committee. Only the members of this committee can merge into master.

